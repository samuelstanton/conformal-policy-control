import gc
import logging
import numpy as np
import os
import s3fs
import tempfile
import time
import torch
import torch.distributed.checkpoint as dist_cp

from finetune_utils import maybe_log
from string import Template
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple, Union

import sys


CUDA_ERROR = getattr(torch.cuda, "CudaError", RuntimeError)

def wait_for_gpu_availability(
    device: Optional[str] = None,
    max_wait_time: int = 172800, # 3600, # Maximum wait time in seconds (48 hour default)
    check_interval: int = 30,  # Check every 30 seconds
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Wait for GPU to become available before proceeding.
    
    Args:
        device: Device string ('cuda', 'cpu', etc.). If None or 'cpu', returns immediately.
        max_wait_time: Maximum time to wait in seconds (default: 3600 = 1 hour)
        check_interval: Time between checks in seconds (default: 5)
        logger: Optional logger for status messages
        
    Returns:
        True if GPU is available, False if max_wait_time exceeded
    """
    # If device is None or 'cpu', no need to wait
    if device is None or device == "cpu":
        return True
    
    # If device is 'gpu', convert to 'cuda'
    if device == "gpu":
        device = "cuda"
    
    # Only wait if device is 'cuda'
    if device != "cuda":
        return True
    
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < max_wait_time:
        try:
            # Try to initialize CUDA and check if it's available
            if torch.cuda.is_available():
                # Try to allocate a small tensor to verify GPU is actually accessible
                try:
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    if logger:
                        logger.info(f"GPU is now available after {attempts * check_interval} seconds")
                    return True
                except (CUDA_ERROR, RuntimeError) as e:
                    # GPU exists but is busy/unavailable
                    if logger:
                        logger.warning(
                            f"GPU detected but busy/unavailable (attempt {attempts + 1}): {e}. "
                            f"Waiting {check_interval} seconds before retry..."
                        )
            else:
                if logger:
                    logger.warning(
                        f"CUDA not available (attempt {attempts + 1}). "
                        f"Waiting {check_interval} seconds before retry..."
                    )
        except Exception as e:
            if logger:
                logger.warning(
                    f"Error checking GPU availability (attempt {attempts + 1}): {e}. "
                    f"Waiting {check_interval} seconds before retry..."
                )
        
        attempts += 1
        time.sleep(check_interval)
    
    # Max wait time exceeded
    if logger:
        logger.error(
            f"GPU did not become available within {max_wait_time} seconds. "
            f"Total attempts: {attempts}"
        )
    return False

class ModelClient:
    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        hf_model_name: str = None,  # The name of the model in the HF hub. Only necessary
        # if loading from a sharded FSDP checkpoint.
        logger: logging.Logger = None,
        temperature: float = 1.0,
        max_generate_length: int = 500,
        device: Optional[str] = None,
        **model_init_args,
    ):
        self.model_name_or_path = model_name_or_path
        self.hf_model_name = hf_model_name
        self.logger = logger

        self.max_generate_length = max_generate_length
        self.temperature = temperature

        self.device = device
        self.model_init_args = model_init_args

        # Determine the actual device to use
        actual_device = device
        if actual_device is None:
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif actual_device == "gpu":
            actual_device = "cuda"

        # Wait for GPU availability if using CUDA
        if actual_device == "cuda":
            maybe_log(
                self.logger,
                "Waiting for GPU to become available...",
                level="info",
            )
            gpu_available = wait_for_gpu_availability(
                device=actual_device,
                max_wait_time=172800,  # Wait up to 48 hours
                check_interval=30, # Check every 30 seconds
                logger=self.logger,
            )
            if not gpu_available:
                maybe_log(
                    self.logger,
                    "GPU did not become available within timeout. Falling back to CPU.",
                    level="warning",
                )
                actual_device = "cpu"
                self.device = "cpu"

        if model_name_or_path.startswith("s3://"):
            self._load_from_s3(model_name_or_path)
        elif (
            os.path.exists(model_name_or_path)
            and not os.path.exists(f"{model_name_or_path}/config.json")
            and os.path.exists(f"{model_name_or_path}/pytorch_model_fsdp_0")
        ):
            maybe_log(
                self.logger,
                f"{model_name_or_path} is a sharded FSDP model. Attempting to consolidate and convert.",
                level="info",
            )
            self.model, self.config, self.tokenizer = self._convert_checkpoint(
                hf_model_name=hf_model_name,
                fsdp_model_path=f"{model_name_or_path}/pytorch_model_fsdp_0",
                output_path=model_name_or_path,
            )
        else:
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            extra_model_kwargs = {**self.model_init_args}
            if "gg-hf/gemma" in self.model_name_or_path:
                extra_model_kwargs["torch_dtype"] = torch.bfloat16
                # Only use device_map="cuda" if we're actually using CUDA
                if actual_device == "cuda":
                    extra_model_kwargs["device_map"] = "cuda"
            
            # Wrap model loading in retry logic for CUDA errors
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name_or_path, trust_remote_code=True, **extra_model_kwargs
                    )
                    break  # Success, exit retry loop
                except (RuntimeError, CUDA_ERROR) as e:
                    error_str = str(e)
                    if "CUDA error" in error_str or "AcceleratorError" in error_str or "busy or unavailable" in error_str:
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = 10 * retry_count  # Exponential backoff
                            maybe_log(
                                self.logger,
                                f"CUDA error during model loading (attempt {retry_count}/{max_retries}): {e}. "
                                f"Waiting {wait_time} seconds before retry...",
                                level="warning",
                            )
                            time.sleep(wait_time)
                            # Wait for GPU availability again
                            if actual_device == "cuda":
                                wait_for_gpu_availability(
                                    device=actual_device,
                                    max_wait_time=172800,  # Wait up to 48 hours for GPU
                                    check_interval=30, # Check every 30 seconds
                                    logger=self.logger,
                                )
                        else:
                            maybe_log(
                                self.logger,
                                f"Failed to load model after {max_retries} attempts. Error: {e}",
                                level="error",
                            )
                            raise
                    else:
                        # Not a CUDA availability error, re-raise immediately
                        raise
            
            if self.device is not None:
                self.model = self.model.to(self.device)
            elif actual_device is not None and actual_device != "cpu":
                # If device was originally None but we determined it should be cuda, move to cuda
                self.model = self.model.to(actual_device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError(
                    f"Tokenizer for {self.model_name_or_path} does not have a PAD token."
                )
        if hasattr(self.config, "max_position_embeddings"):
            self.max_length = self.config.max_position_embeddings
        elif hasattr(self.config, "max_sequence_length"):
            self.max_length = self.config.max_sequence_length
        elif hasattr(self.config, "n_positions"):
            self.max_length = self.config.n_positions
        else:
            raise ValueError(
                f"Could not find max_position_embeddings, n_positions, or max_sequence_length in model config for {self.model_name_or_path}"
            )

    def _load_from_s3(self, s3_uri: str):
        # Determine the actual device to use
        actual_device = self.device
        if actual_device is None:
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif actual_device == "gpu":
            actual_device = "cuda"

        # Wait for GPU availability if using CUDA
        if actual_device == "cuda":
            gpu_available = wait_for_gpu_availability(
                device=actual_device,
                max_wait_time=172800,  # Wait up to 48 hours
                check_interval=30, # Check every 30 seconds
                logger=self.logger,
            )
            if not gpu_available:
                maybe_log(
                    self.logger,
                    "GPU did not become available within timeout. Falling back to CPU.",
                    level="warning",
                )
                actual_device = "cpu"
                self.device = "cpu"

        self.s3 = s3fs.S3FileSystem()
        with tempfile.TemporaryDirectory() as td:
            maybe_log(self.logger, f"Downloading model from {s3_uri} into {td}.")
            if not td.endswith("/"):
                td += "/"
            if not s3_uri.endswith("/"):
                s3_uri += "/"
            s3_uri += (
                "*"  # will copy only the files in the directory, not any nested ones
            )
            self.s3.get(s3_uri, td)
            self.config = AutoConfig.from_pretrained(td)
            self.model_init_args["trust_remote_code"] = True
            
            # Wrap model loading in retry logic for CUDA errors
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        td, **self.model_init_args
                    )
                    break  # Success, exit retry loop
                except (RuntimeError, CUDA_ERROR) as e:
                    error_str = str(e)
                    if "CUDA error" in error_str or "AcceleratorError" in error_str or "busy or unavailable" in error_str:
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = 10 * retry_count  # Exponential backoff
                            maybe_log(
                                self.logger,
                                f"CUDA error during model loading from S3 (attempt {retry_count}/{max_retries}): {e}. "
                                f"Waiting {wait_time} seconds before retry...",
                                level="warning",
                            )
                            time.sleep(wait_time)
                            # Wait for GPU availability again
                            if actual_device == "cuda":
                                wait_for_gpu_availability(
                                    device=actual_device,
                                    max_wait_time=172800,
                                    check_interval=30, # Check every 30 seconds
                                    # max_wait_time=300,  # Wait up to 5 minutes per retry
                                    # check_interval=5,
                                    logger=self.logger,
                                )
                        else:
                            maybe_log(
                                self.logger,
                                f"Failed to load model from S3 after {max_retries} attempts. Error: {e}",
                                level="error",
                            )
                            raise
                    else:
                        # Not a CUDA availability error, re-raise immediately
                        raise
            
            if self.device is not None:
                self.model = self.model.to(self.device)
            elif actual_device is not None and actual_device != "cpu":
                self.model = self.model.to(actual_device)
            self.tokenizer = AutoTokenizer.from_pretrained(td)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _convert_checkpoint(
        self, hf_model_name: str, fsdp_model_path: str, output_path: str
    ):
        """
        hf_model_name: Name of model in HF Hub, e.g. "gpt2".
        fsdp_model_path: path to the fsdp checkpoint, for example `/x/checkpoint-xxx/pytorch_model_fsdp_x`
        output_path: output path to save the converted checkpoint
        """
        # Determine the actual device to use
        actual_device = self.device
        if actual_device is None:
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif actual_device == "gpu":
            actual_device = "cuda"

        # Wait for GPU availability if using CUDA (required for checkpoint conversion)
        if actual_device == "cuda":
            gpu_available = wait_for_gpu_availability(
                device=actual_device,
                max_wait_time=172800, #3600,  # Wait up to 48 hours
                check_interval=30,
                logger=self.logger,
            )
            if not gpu_available:
                maybe_log(
                    self.logger,
                    "GPU did not become available within timeout. Checkpoint conversion requires GPU.",
                    level="error",
                )
                raise RuntimeError("GPU required for checkpoint conversion but not available")

        config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        
        # Wrap model creation in retry logic for CUDA errors
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True, **self.model_init_args
                ).cuda()
                break  # Success, exit retry loop
            except (RuntimeError, CUDA_ERROR) as e:
                error_str = str(e)
                if "CUDA error" in error_str or "AcceleratorError" in error_str or "busy or unavailable" in error_str:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 10 * retry_count  # Exponential backoff
                        maybe_log(
                            self.logger,
                            f"CUDA error during checkpoint conversion (attempt {retry_count}/{max_retries}): {e}. "
                            f"Waiting {wait_time} seconds before retry...",
                            level="warning",
                        )
                        time.sleep(wait_time)
                        # Wait for GPU availability again
                        wait_for_gpu_availability(
                            device="cuda",
                            max_wait_time=172800, #300,  
                            check_interval=30,
                            logger=self.logger,
                        )
                    else:
                        maybe_log(
                            self.logger,
                            f"Failed to convert checkpoint after {max_retries} attempts. Error: {e}",
                            level="error",
                        )
                        raise
                else:
                    # Not a CUDA availability error, re-raise immediately
                    raise
        
        model = self._load_sharded_model_single_gpu(model, fsdp_model_path)
        model.save_pretrained(output_path, max_shard_size="10GB")
        tokenizer.save_pretrained(output_path)
        return model, config, tokenizer

    def _load_sharded_model_single_gpu(self, model, model_path):
        state_dict = {"model": model.state_dict()}

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(model_path),
            no_dist=True,
        )

        result = model.load_state_dict(state_dict["model"])
        maybe_log(
            self.logger,
            f"Sharded state checkpoint loaded from {model_path}. Result: {result}",
            level="info",
        )
        return model

    def _apply_chat_template(self, msgs) -> str:
        if self.tokenizer.chat_template is not None:
            chat_msgs = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            raise NotImplementedError(
                f"Chat template does not exist for model {self.model_name_or_path}."
            )
        return chat_msgs

    def _tokenize_batch(
        self,
        text_batch: Union[List[List[Dict[str, str]]], List[str]],
        max_generate_length: Optional[int] = None,
    ) -> torch.tensor:
        """Tokenize either a batch of messages or a single text string."""
        if not text_batch:
            return None
        if max_generate_length is not None:
            max_gen_length = max_generate_length
        else:
            max_gen_length = self.max_generate_length
        max_context_length = self.max_length - max_gen_length
        if isinstance(text_batch[0], list):
            # text batch is list of messages, so apply chat template
            texts_to_tokenize = [self._apply_chat_template(text) for text in text_batch]
        else:
            texts_to_tokenize = text_batch
        tokenized = self.tokenizer(
            texts_to_tokenize,
            truncation=True,
            padding="longest",
            max_length=max_context_length,
            return_tensors="pt",
        )
        return tokenized

    def _chat_hf_model(
        self, msgs: Union[List[Dict[str, str]], str], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion for either a series of messages or a single text prompt.
        """
        tokenized = self._tokenize_batch([msgs]).input_ids
        if self.device is not None:
            tokenized = tokenized.to(self.device)
        outputs = self.model.generate(tokenized, **kwargs)
        return {"output_strs": self.tokenizer.batch_decode(outputs)}

    def chat_single_turn(self, msgs: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return self._chat_hf_model(msgs, **kwargs)

    def chat_raw_logits(
        self, input_ids: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        """Alternative API for getting all logits."""
        return self.model(input_ids, attention_mask=attention_mask).logits

    def chat_single_turn_text(self, text: str, **kwargs) -> Dict[str, Any]:
        msgs = [{"role": "user", "content": text}]
        return self._chat_hf_model(msgs, **kwargs)



    ### NOTE: Had attempted modifying this function for unconditional generation, wasn't working so reverted to 
    ### original version below this commented-out block
    # @torch.no_grad()
    # def generate_texts_batched(
    #     self,
    #     input_texts: Optional[List[str]] = None,
    #     batch_size: int = 8,
    #     return_likelihoods=False,
    #     subsample_seeds=False,
    #     **kwargs,
    # ) -> Union[List[str], Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]]:
    #     """
    #     Runs batched generation. If return_likelihoods=True, returns a tuple containing
    #     the output sequences, the output tokens, and the output token log-likelihoods.
    #     Otherwise, returns only the output sequences.

    #     Args:
    #         input_texts: List of input texts for conditional generation. If None, generates
    #                     sequences unconditionally starting from BOS token or other appropriate
    #                     start token (CLS, PAD, or space as fallback).
    #         batch_size: Number of sequences to generate per batch
    #         return_likelihoods: Whether to return token likelihoods
    #         subsample_seeds: Whether to subsample input seeds with replacement
    #         **kwargs: Additional generation arguments passed to model.generate()

    #     if subsamples_seeds:
    #         Subsample input seeds uniformly at random, with replacement, from input_texts.
    #     """
    #     if input_texts is None:
    #         # Unconditional generation - create proper BOS tokens
    #         if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
    #             # Use beginning-of-sequence token if available
    #             input_texts = [self.tokenizer.bos_token] * batch_size
    #             if self.logger:
    #                 self.logger.info(f"Using BOS token '{self.tokenizer.bos_token}' for unconditional generation")
    #         else:
    #             # Fallback: use a minimal context token that won't cause embedding issues
    #             # Most tokenizers have a special token we can use
    #             if hasattr(self.tokenizer, 'cls_token') and self.tokenizer.cls_token:
    #                 input_texts = [self.tokenizer.cls_token] * batch_size
    #                 if self.logger:
    #                     self.logger.info(f"Using CLS token '{self.tokenizer.cls_token}' for unconditional generation")
    #             elif hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token:
    #                 input_texts = [self.tokenizer.pad_token] * batch_size
    #                 if self.logger:
    #                     self.logger.info(f"Using PAD token '{self.tokenizer.pad_token}' for unconditional generation")
    #             else:
    #                 # Last resort: use a single space character which should tokenize properly
    #                 input_texts = [" "] * batch_size
    #                 if self.logger:
    #                     self.logger.info("Using space character for unconditional generation (fallback)")
    #         is_unconditional = True
    #     else:
    #         is_unconditional = False
        
    #     output_strs = []
    #     all_output_token_ids = []
    #     all_token_logps = []
    #     for batch_start_idx in tqdm(
    #         range(0, len(input_texts), batch_size), desc="Batched generation..."
    #     ):  
    #         if subsample_seeds:
    #             ## In this condition: Subsample input seed seqs uniformly with replacement 
    #             ## (ensures that each output is sampled from the mixture of the prompt-conditional distributions)
    #             idx_input_texts = np.random.choice(range(len(input_texts)), size=batch_size, replace=True)
    #             batch_texts = [input_texts[idx] for idx in idx_input_texts]
    #         else:
    #             ## In this condition: Move through all the input seeds in minibatches
    #             batch_end_idx = batch_start_idx + batch_size
    #             batch_texts = input_texts[batch_start_idx:batch_end_idx]

    #         input_ids = self._tokenize_batch(batch_texts).input_ids
    #         if self.device is not None:
    #             input_ids = input_ids.to(self.device)
    #         gen_args = {**kwargs}
    #         if return_likelihoods:
    #             gen_args["return_dict_in_generate"] = True
    #             gen_args["output_scores"] = True
    #         outputs = self.model.generate(
    #             input_ids, **gen_args, tokenizer=self.tokenizer
    #         )
    #         output_token_ids = outputs.sequences if return_likelihoods else outputs
    #         self.logger.info(f"len(output_token_ids) '{len(output_token_ids)}', output_token_ids[0] : {output_token_ids[0]}")
            
    #         # Handle output decoding based on generation type
    #         if is_unconditional:
    #             # For unconditional generation, remove the BOS/start token to get clean sequences
    #             if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
    #                 # Remove the BOS token from output
    #                 decoded_outputs = self.tokenizer.batch_decode(
    #                     output_token_ids[:, 1:],  # Skip BOS token
    #                     skip_special_tokens=True,
    #                     clean_up_tokenization_spaces=True,
    #                 )
    #             else:
    #                 # Keep the full sequence if no BOS token
    #                 decoded_outputs = self.tokenizer.batch_decode(
    #                     output_token_ids,
    #                     skip_special_tokens=True,
    #                     clean_up_tokenization_spaces=True,
    #                 )
    #         else:
    #             # For conditional generation, remove the input context
    #             decoded_outputs = self.tokenizer.batch_decode(
    #                 output_token_ids[:, input_ids.shape[-1] :],
    #                 skip_special_tokens=True,
    #                 clean_up_tokenization_spaces=True,
    #             )
    #         output_strs.extend(decoded_outputs)
    #         if return_likelihoods:
    #             token_logps = self.model.compute_transition_scores(
    #                 outputs.sequences, outputs.scores, normalize_logits=True
    #             )
    #             # Handle token extraction based on generation type
    #             if is_unconditional:
    #                 if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
    #                     # Remove the BOS token for likelihood computation
    #                     for oti in output_token_ids[:, 1:]:
    #                         all_output_token_ids.append(oti)
    #                 else:
    #                     # Keep the full sequence if no BOS token
    #                     for oti in output_token_ids:
    #                         all_output_token_ids.append(oti)
    #             else:
    #                 for oti in output_token_ids[:, input_ids.shape[-1] :]:
    #                     all_output_token_ids.append(oti)
    #             for o_logps in token_logps:
    #                 all_token_logps.append(o_logps)
    #     if return_likelihoods:
    #         return (output_strs, all_output_token_ids, all_token_logps)
    #     return output_strs
    


    @torch.no_grad()
    def generate_texts_batched(
        self,
        input_texts: List[str],
        batch_size: int = 8,
        return_likelihoods=False,
        subsample_seeds=False,
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]]:
        """
        Runs batched generation. If return_likelihoods=True, returns a tuple containing
        the output sequences, the output tokens, and the output token log-likelihoods.
        Otherwise, returns only the output sequences.
        """
        output_strs = []
        all_output_token_ids = []
        all_token_logps = []
        for batch_start_idx in tqdm(
            range(0, len(input_texts), batch_size), desc="Batched generation..."
        ):  
            if subsample_seeds:
                ## In this condition: Subsample input seed seqs uniformly with replacement 
                ## (ensures that each output is sampled from the mixture of the prompt-conditional distributions)
                idx_input_texts = np.random.choice(range(len(input_texts)), size=batch_size, replace=True)
                batch_texts = [input_texts[idx] for idx in idx_input_texts]
            else:
                ## In this condition: Move through all the input seeds in minibatches
                batch_end_idx = batch_start_idx + batch_size
                batch_texts = input_texts[batch_start_idx:batch_end_idx]

            input_ids = self._tokenize_batch(batch_texts).input_ids
            if self.device is not None:
                input_ids = input_ids.to(self.device)
            gen_args = {**kwargs}
            if return_likelihoods:
                gen_args["return_dict_in_generate"] = True
                gen_args["output_scores"] = True
            outputs = self.model.generate(
                input_ids, **gen_args, tokenizer=self.tokenizer
            )
            output_token_ids = outputs.sequences if return_likelihoods else outputs
            # remove the input from each output
            decoded_outputs = self.tokenizer.batch_decode(
                output_token_ids[:, input_ids.shape[-1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # logger.info(f"")
            output_strs.extend(decoded_outputs)
            if return_likelihoods:
                token_logps = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                for oti in output_token_ids[:, input_ids.shape[-1] :]:
                    all_output_token_ids.append(oti)
                for o_logps in token_logps:
                    all_token_logps.append(o_logps)
        if return_likelihoods:
            return (output_strs, all_output_token_ids, all_token_logps)
        return output_strs




    @torch.no_grad()
    def compute_likelihoods(
        self,
        inputs: List[str],
        targets: List[str],
        batch_size: int = 16,
        device: str = "cuda",
        # add_start_token: bool = True,
        logger: logging.Logger = None,
    ) -> List[float]:
        # Compute length-normalized likelihoods of target tokens given the input
        # TODO: write test to check that the logprobs match what model.generate and compute_transition_scores gives!
        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
            ], "device should be in ['gpu', 'cpu', 'cuda']."
            if device == "gpu":
                device = "cuda"
        else:
            device = self.device

        all_likelihoods = []
        for start_index in tqdm(
            range(0, len(inputs), batch_size), desc="Computing log likelihood..."
        ):
            end_index = min(start_index + batch_size, len(inputs))
            batch = [
                f"{input}{target}"
                for input, target in zip(
                    inputs[start_index:end_index], targets[start_index:end_index]
                )
            ]
            tokenized = self._tokenize_batch(batch, max_generate_length=0).to(
                self.device
            )
            outputs = self.model(**tokenized)
            scores = outputs.logits
            # probs = torch.exp(torch.nn.functional.log_softmax(scores, dim=-1))

            ### NOTE: Think this should be log_softmax (without exp as above)
            probs = torch.nn.functional.log_softmax(scores, dim=-1) ###
            # logger.info(f"probs shape : {probs.shape}")
            # get probs and mask out both padding and input tokens
            inputs_tokenized = self._tokenize_batch(inputs[start_index:end_index])
            input_seq_lens = inputs_tokenized.attention_mask.sum(-1)
            input_tokens_mask = torch.LongTensor(
                [
                    [0 for _ in range(input_seq_lens[i].item())]
                    + [
                        1
                        for _ in range(
                            tokenized.input_ids.shape[1] - input_seq_lens[i].item()
                        )
                    ]
                    for i in range(tokenized.input_ids.shape[0])
                ]
            ).to(self.device)
            indexes = tokenized.input_ids[:, 1:]
            next_token_probs = (
                torch.gather(probs, -1, indexes.unsqueeze(-1)).squeeze(-1)
                * tokenized.attention_mask[:, 1:]
                * input_tokens_mask[:, 1:]
            )

            next_token_probs = next_token_probs.cpu().numpy()

            targets_tokenized = self._tokenize_batch(targets[start_index:end_index])
            targets_seq_lens = targets_tokenized.attention_mask.sum(-1).cpu().numpy()
            avg_likelihoods = list(next_token_probs.sum(-1) / targets_seq_lens)
            # logger.info(f"avg_likelihoods shape : {np.shape(avg_likelihoods)}")
            # logger.info(f"seq likelihood : {list(next_token_probs.prod(-1) / targets_seq_lens)}")
            
            all_likelihoods.extend(avg_likelihoods)
            # logger.info(f"all_likelihoods : {all_likelihoods}")

        return all_likelihoods





    @torch.no_grad()
    def compute_likelihoods_avg(
        self,
        inputs: List[str],
        targets: List[str],
        batch_size: int = 10,
        device: str = "cuda",
        float_constant: int = 10**10, ## constant to scale probabilities by to reduce float point issues
        # add_start_token: bool = True,
        logger: logging.Logger = None,
    ) -> List[float]:
        """
        Compute likelihoods of target sequences, averaged over all provided seeds
        """

        logger.info(f"temperature : {self.temperature}")

        # Compute length-normalized likelihoods of target tokens given the input
        # TODO: write test to check that the logprobs match what model.generate and compute_transition_scores gives!
        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
            ], "device should be in ['gpu', 'cpu', 'cuda']."
            if device == "gpu":
                device = "cuda"
        else:
            device = self.device

        # all_likelihoods = []
        all_likelihoods_torch = torch.zeros(len(targets), dtype=torch.float64)

        ## Looping over target sequences
        for t, target in enumerate(tqdm(targets, desc="Computing log likelihoods averaged over inputs...")):
            
            log_likelihoods_target = []
            
            ## Looping over batches of input sequences
            for start_index in range(0, len(inputs), batch_size):  
                
                end_index = min(start_index + batch_size, len(inputs))

                ## For a given target, minibatch of concatenations with subset of input prompt sequences
                batch = [f"{input}{target}" for input in inputs[start_index:end_index]]

                tokenized = self._tokenize_batch(batch, max_generate_length=0).to(
                    self.device
                )
                outputs = self.model(**tokenized)
                scores = outputs.logits / self.temperature
                

                # probs = torch.exp(torch.nn.functional.log_softmax(scores, dim=-1)) ## Seems unnecessary to do exp(log(softmax)) ??
                probs = torch.nn.functional.log_softmax(scores, dim=-1)

                # probs = torch.clip(torch.nn.functional.softmax(scores, dim=-1), min=sys.float_info.min) ## Clip probabilities at minimum float value



                # logger.info(f"torch.nn.functional.softmax(scores, dim=-1) : {torch.nn.functional.softmax(scores, dim=-1).shape}")
                # logger.info(f"clipped probs : {probs.shape}")

                # # probs = torch.log(scores)

                # logger.info(f"log clipped probs : {probs.shape}")


                ## TO DO: Compute probabilities averaged over input sequences

                # get probs and mask out both padding and input tokens
                inputs_tokenized = self._tokenize_batch(inputs[start_index:end_index])
                input_seq_lens = inputs_tokenized.attention_mask.sum(-1) ## length = 75 
                # logger.info(f"input_seq_lens : {input_seq_lens}")
                input_tokens_mask = torch.LongTensor(
                    [
                        [0 for _ in range(input_seq_lens[i].item())]
                        + [
                            1
                            for _ in range(
                                tokenized.input_ids.shape[1] - input_seq_lens[i].item()
                            )
                        ]
                        for i in range(tokenized.input_ids.shape[0])
                    ]
                ).to(self.device)
                indexes = tokenized.input_ids[:, 1:]
                # logger.info(f"probs.shape : {probs.shape}")
                # # logger.info(f"probs : {probs}")
                # logger.info(f"indexes.unsqueeze(-1).shape : {indexes.unsqueeze(-1).shape}")
                # # logger.info(f"indexes.unsqueeze(-1).shape : {indexes.unsqueeze(-1).shape}")
                # logger.info(f"probs[:,:,:100] : {probs[:,:,:100]}")
                # logger.info(f"torch.sum(exp(probs), dim=2) : {torch.sum(np.exp(probs), dim=2)}")


                # logger.info(f"torch.gather(probs, -1, indexes.unsqueeze(-1)).shape : {torch.gather(probs, -1, indexes.unsqueeze(-1))[:, 74:84]}")
                # logger.info(f"tokenized.attention_mask[:, 1:] : {tokenized.attention_mask[:, 74:84]}")
                # logger.info(f"input_tokens_mask[:, 1:] : {input_tokens_mask[:, 74:84]}")
                # logger.info(f"torch.gather(probs, -1, indexes.unsqueeze(-1)) : {torch.gather(probs, -1, indexes.unsqueeze(-1))}")

                next_token_probs = (
                    torch.gather(probs, -1, indexes.unsqueeze(-1)).squeeze(-1)
                    * tokenized.attention_mask[:, 1:]
                    * input_tokens_mask[:, 1:]
                )
                
                next_token_probs = next_token_probs.cpu().numpy()
                targets_tokenized = self._tokenize_batch(targets[start_index:end_index])
                targets_seq_lens = targets_tokenized.attention_mask.sum(-1).cpu().numpy()  ## length = 75 
                # logger.info(f"targets_seq_lens : {targets_seq_lens}")
                log_likelihoods_batch = list(next_token_probs.sum(-1)) # / targets_seq_lens)

                # logger.info(f"start_index : {start_index}, end_index : {end_index}")
                # logger.info(f"next_token_probs : {next_token_probs}")

                # logger.info(f"next_token_probs.sum(-1) : {next_token_probs.sum(-1)}")
                # logger.info(f"targets_seq_lens : {targets_seq_lens}")
                # logger.info(f"sum log_likelihoods_batch      : {log_likelihoods_batch}")
                # logger.info(f"likelihoods_batch      : {np.exp(log_likelihoods_batch)}")


                # logger.info("\n\n\n")
                # logger.info(f"next_token_probs.shape : {next_token_probs.shape}")
                # # logger.info(f"next_token_probs       : {next_token_probs}")
                # logger.info(f"sum log_liks next token       : {np.sum(next_token_probs[:, 74:])}")
                # logger.info(f"exp sum log_liks next token   : {np.exp(np.sum(next_token_probs[:, 74:]))}")
                # logger.info(f"input_seq_lens : {input_seq_lens}")
                # logger.info(f"prod next_token_probs       : {np.prod(next_token_probs[input_seq_lens:end_index])}")




                log_likelihoods_target.extend(log_likelihoods_batch)

                # seq_log_likelihoods = list(next_token_probs.prod(-1) / targets_seq_lens)
                # all_log_likelihoods.extend(avg_log_likelihoods)
            #     logger.info(f"likelihoods_target len : {len(log_likelihoods_target)}")
            # log_likelihoods_target_scaled = np.array(log_likelihoods_target) #/ self.max_generate_length
            # logger.info(f"likelihoods_target : {log_likelihoods_target}")
            # logger.info(f"self.max_generate_length : {self.max_generate_length}")
            # logger.info(f"log_likelihoods_target : {log_likelihoods_target}")
            # likelihoods_target_scaled = np.exp(log_likelihoods_target) * float_constant  ## Scale up by constant to reduce float point issues
            # logger.info(f"likelihoods_target : {np.exp(log_likelihoods_target)}")
            # logger.info(f"likelihoods_target_scaled : {log_likelihoods_target_scaled}")
            # logger.info(f"likelihoods_scaled_avg    : {max(np.exp(log_likelihoods_target_scaled).mean(), sys.float_info.min)}")
            # all_likelihoods.append(max(np.exp(log_likelihoods_target_scaled).mean(), sys.float_info.min)) ## Clip likelihoods at minimum float value for now (if was generated, then actual likelihood is positive)
            # logger.info(f"log_likelihoods_target : {log_likelihoods_target}")
            # logger.info(f"exp(log_likelihoods_target)   : {torch.exp(torch.tensor(log_likelihoods_target, dtype=torch.float64))}")
            
            # all_likelihoods.append(torch.exp(torch.tensor(log_likelihoods_target, dtype=torch.float64)).mean()) ## Clip likelihoods at minimum float value for now (if was generated, then actual likelihood is positive)
            all_likelihoods_torch[t] = torch.exp(torch.tensor(log_likelihoods_target, dtype=torch.float64)).mean()

            # all_likelihoods.append(torch.mean(log_likelihoods_target))
            # all_likelihoods.append(likelihoods_target.prod())


            # logger.info(f"np.exp(log_likelihoods_target) : {np.exp(log_likelihoods_target)}")
            # logger.info(f"liks prior to avg : {np.exp(log_likelihoods_target_scaled)}")
            # logger.info(f"seq likelihood : {all_likelihoods[-1]}")
            # logger.info(f"all_likelihoods : {all_likelihoods}")
            # logger.info(f"len(all_likelihoods) : {len(all_likelihoods)}")

            # if t == 2:
            #     raise ValueError("Stopping for debugging")

                
        return all_likelihoods_torch.tolist()







    ### NOTE: Had attempted modifying this function for unconditional generation, wasn't working so reverted to 
    ### original version below this commented-out block
    # @torch.no_grad()
    # def generate_texts_batched(
    #     self,
    #     input_texts: Optional[List[str]] = None,
    #     batch_size: int = 8,
    #     return_likelihoods=False,
    #     subsample_seeds=False,
    #     **kwargs,
    # ) -> Union[List[str], Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]]:
    #     """
    #     Runs batched generation. If return_likelihoods=True, returns a tuple containing
    #     the output sequences, the output tokens, and the output token log-likelihoods.
    #     Otherwise, returns only the output sequences.

    #     Args:
    #         input_texts: List of input texts for conditional generation. If None, generates
    #                     sequences unconditionally starting from BOS token or other appropriate
    #                     start token (CLS, PAD, or space as fallback).
    #         batch_size: Number of sequences to generate per batch
    #         return_likelihoods: Whether to return token likelihoods
    #         subsample_seeds: Whether to subsample input seeds with replacement
    #         **kwargs: Additional generation arguments passed to model.generate()

    #     if subsamples_seeds:
    #         Subsample input seeds uniformly at random, with replacement, from input_texts.
    #     """
    #     if input_texts is None:
    #         # Unconditional generation - create proper BOS tokens
    #         if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
    #             # Use beginning-of-sequence token if available
    #             input_texts = [self.tokenizer.bos_token] * batch_size
    #             if self.logger:
    #                 self.logger.info(f"Using BOS token '{self.tokenizer.bos_token}' for unconditional generation")
    #         else:
    #             # Fallback: use a minimal context token that won't cause embedding issues
    #             # Most tokenizers have a special token we can use
    #             if hasattr(self.tokenizer, 'cls_token') and self.tokenizer.cls_token:
    #                 input_texts = [self.tokenizer.cls_token] * batch_size
    #                 if self.logger:
    #                     self.logger.info(f"Using CLS token '{self.tokenizer.cls_token}' for unconditional generation")
    #             elif hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token:
    #                 input_texts = [self.tokenizer.pad_token] * batch_size
    #                 if self.logger:
    #                     self.logger.info(f"Using PAD token '{self.tokenizer.pad_token}' for unconditional generation")
    #             else:
    #                 # Last resort: use a single space character which should tokenize properly
    #                 input_texts = [" "] * batch_size
    #                 if self.logger:
    #                     self.logger.info("Using space character for unconditional generation (fallback)")
    #         is_unconditional = True
    #     else:
    #         is_unconditional = False
        
    #     output_strs = []
    #     all_output_token_ids = []
    #     all_token_logps = []
    #     for batch_start_idx in tqdm(
    #         range(0, len(input_texts), batch_size), desc="Batched generation..."
    #     ):  
    #         if subsample_seeds:
    #             ## In this condition: Subsample input seed seqs uniformly with replacement 
    #             ## (ensures that each output is sampled from the mixture of the prompt-conditional distributions)
    #             idx_input_texts = np.random.choice(range(len(input_texts)), size=batch_size, replace=True)
    #             batch_texts = [input_texts[idx] for idx in idx_input_texts]
    #         else:
    #             ## In this condition: Move through all the input seeds in minibatches
    #             batch_end_idx = batch_start_idx + batch_size
    #             batch_texts = input_texts[batch_start_idx:batch_end_idx]

    #         input_ids = self._tokenize_batch(batch_texts).input_ids
    #         if self.device is not None:
    #             input_ids = input_ids.to(self.device)
    #         gen_args = {**kwargs}
    #         if return_likelihoods:
    #             gen_args["return_dict_in_generate"] = True
    #             gen_args["output_scores"] = True
    #         outputs = self.model.generate(
    #             input_ids, **gen_args, tokenizer=self.tokenizer
    #         )
    #         output_token_ids = outputs.sequences if return_likelihoods else outputs
    #         self.logger.info(f"len(output_token_ids) '{len(output_token_ids)}', output_token_ids[0] : {output_token_ids[0]}")
            
    #         # Handle output decoding based on generation type
    #         if is_unconditional:
    #             # For unconditional generation, remove the BOS/start token to get clean sequences
    #             if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
    #                 # Remove the BOS token from output
    #                 decoded_outputs = self.tokenizer.batch_decode(
    #                     output_token_ids[:, 1:],  # Skip BOS token
    #                     skip_special_tokens=True,
    #                     clean_up_tokenization_spaces=True,
    #                 )
    #             else:
    #                 # Keep the full sequence if no BOS token
    #                 decoded_outputs = self.tokenizer.batch_decode(
    #                     output_token_ids,
    #                     skip_special_tokens=True,
    #                     clean_up_tokenization_spaces=True,
    #                 )
    #         else:
    #             # For conditional generation, remove the input context
    #             decoded_outputs = self.tokenizer.batch_decode(
    #                 output_token_ids[:, input_ids.shape[-1] :],
    #                 skip_special_tokens=True,
    #                 clean_up_tokenization_spaces=True,
    #             )
    #         output_strs.extend(decoded_outputs)
    #         if return_likelihoods:
    #             token_logps = self.model.compute_transition_scores(
    #                 outputs.sequences, outputs.scores, normalize_logits=True
    #             )
    #             # Handle token extraction based on generation type
    #             if is_unconditional:
    #                 if hasattr(self.tokenizer, 'bos_token') and self.tokenizer.bos_token:
    #                     # Remove the BOS token for likelihood computation
    #                     for oti in output_token_ids[:, 1:]:
    #                         all_output_token_ids.append(oti)
    #                 else:
    #                     # Keep the full sequence if no BOS token
    #                     for oti in output_token_ids:
    #                         all_output_token_ids.append(oti)
    #             else:
    #                 for oti in output_token_ids[:, input_ids.shape[-1] :]:
    #                     all_output_token_ids.append(oti)
    #             for o_logps in token_logps:
    #                 all_token_logps.append(o_logps)
    #     if return_likelihoods:
    #         return (output_strs, all_output_token_ids, all_token_logps)
    #     return output_strs
    



    # @torch.no_grad()
    # def generate_texts_batched_contrastive_mixture(
    #     self,
    #     reference_models: List["ModelClient"],          # List of reference models
    #     input_texts_list: List[List[str]],              # [prompts_B1, ..., prompts_Bk, prompts_A]
    #     # contrastive_proportion: float = 1.0,                             # Weight for log(P_A / P_mix)
    #     # ref_proportion: float = 0.0,                              # Weight for log(P_mix)
    #     # temperature: float = 1.0,
    #     # max_new_tokens: int = 100,
    #     return_likelihood_ratios: bool = False,
    #     mixture_weights: Optional[List[float]] = None,  # Optional weights for mixture (default uniform)
    #     **kwargs,
    # ) -> Union[List[str], Tuple[List[str], List[float]]]:
    #     """
    #     Contrastive decoding against a mixture of reference models where both the target model (A)
    #     and each reference model (Bi) compute next-token probabilities averaged over their own prompt sets.
    #     Uses log( mean(P) ) rather than mean( log(P) ).
    #     """
    #     num_ref_models = len(reference_models)
    #     if num_ref_models == 0:
    #         raise ValueError("At least one reference model must be provided")
    #     if len(input_texts_list) != num_ref_models + 1:
    #         raise ValueError(f"input_texts_list must have length num_ref_models + 1 (got {len(input_texts_list)})")

    #     target_input_texts = input_texts_list[-1]
    #     if len(target_input_texts) == 0:
    #         raise ValueError("Target prompt list must be non-empty")

    #     # Optional: enforce equal counts; remove if variable counts are acceptable
    #     num_sequences = len(target_input_texts)
    #     for i, ref_prompts in enumerate(input_texts_list[:-1]):
    #         if len(ref_prompts) != num_sequences:
    #             raise ValueError(f"input_texts_list[{i}] length ({len(ref_prompts)}) must match target prompts length ({num_sequences})")

    #     # Mixture weights (normalize)
    #     if mixture_weights is None:
    #         mixture_weights = [1.0 / num_ref_models] * num_ref_models
    #     else:
    #         if len(mixture_weights) != num_ref_models:
    #             raise ValueError("mixture_weights length must match number of reference models")
    #         total_w = sum(mixture_weights)
    #         if total_w <= 0:
    #             raise ValueError("mixture_weights must sum to a positive value")
    #         mixture_weights = [w / total_w for w in mixture_weights]

    #     # Tokenize all prompts once
    #     target_input_ids = self._tokenize_batch(target_input_texts).input_ids
    #     if self.device is not None:
    #         target_input_ids = target_input_ids.to(self.device)

    #     ref_input_ids_list = []
    #     for ref_prompts in input_texts_list[:-1]:
    #         ref_ids = self._tokenize_batch(ref_prompts).input_ids
    #         if self.device is not None:
    #             ref_ids = ref_ids.to(self.device)
    #         ref_input_ids_list.append(ref_ids)

    #     # Shared generated suffix across all prompts
    #     generated_suffix = torch.empty(
    #         (1, 0), dtype=target_input_ids.dtype, device=target_input_ids.device
    #     )

    #     per_step_log_ratios: List[float] = []
    #     eps = 1e-12  # numerical stability

    #     for _ in tqdm(range(max_new_tokens), desc="Contrastive generation (avg probs)..."):
    #         # Target: compute probabilities for each prompt, then average probs (not logs)
    #         suffix_repeated_target = generated_suffix.repeat(target_input_ids.size(0), 1)  # [N_target, T]
    #         target_batch = torch.cat([target_input_ids, suffix_repeated_target], dim=1)    # [N_target, L_t+T]
    #         outputs_a = self.model(target_batch)
    #         logits_a_all = outputs_a.logits[:, -1, :]                                       # [N_target, V]
    #         probs_a_all = torch.softmax(logits_a_all / temperature, dim=-1)                 # [N_target, V]
    #         probs_a_avg = probs_a_all.mean(0, keepdim=True)                                  # [1, V]
    #         log_probs_a = torch.log(probs_a_avg.clamp_min(eps))                              # [1, V]

    #         # References: average probs per model over its prompts, then weighted sum across models
    #         mixture_probs = None  # [1, V]
    #         for w, (ref_ids, ref_model) in zip(mixture_weights, zip(ref_input_ids_list, reference_models)):
    #             suffix_repeated_ref = generated_suffix.repeat(ref_ids.size(0), 1)           # [N_ref_i, T]
    #             ref_batch = torch.cat([ref_ids, suffix_repeated_ref], dim=1)                # [N_ref_i, L_i+T]
    #             ref_out = ref_model.model(ref_batch)
    #             ref_logits_all = ref_out.logits[:, -1, :]                                   # [N_ref_i, V]
    #             ref_probs_all = torch.softmax(ref_logits_all / temperature, dim=-1)         # [N_ref_i, V]
    #             ref_probs_avg = ref_probs_all.mean(0, keepdim=True)                          # [1, V]
    #             weighted = w * ref_probs_avg
    #             mixture_probs = weighted if mixture_probs is None else (mixture_probs + weighted)

    #         mixture_log_probs = torch.log(mixture_probs.clamp_min(eps))                      # [1, V]

    #         # Contrastive score per token
    #         # contrastive_scores = contrastive_proportion * (log_probs_a - mixture_log_probs) + ref_proportion * mixture_log_probs  # [1, V]
    #         contrastive_scores = log_probs_a - mixture_log_probs  # [1, V]

    #         # Greedy next token (change to sampling if desired)
    #         next_token = torch.argmax(contrastive_scores, dim=-1, keepdim=True)  # [1,1]

    #         # Track per-step log-ratio if requested
    #         if return_likelihood_ratios:
    #             step_lr = torch.gather(log_probs_a, -1, next_token).squeeze(-1) - torch.gather(mixture_log_probs, -1, next_token).squeeze(-1)
    #             per_step_log_ratios.append(step_lr.item())

    #         # Append to the shared suffix
    #         generated_suffix = torch.cat([generated_suffix, next_token], dim=1)

    #         # Early stop on EOS
    #         if next_token.item() == self.tokenizer.eos_token_id:
    #             break

    #     # Decode using first prompt to strip context; only the suffix is returned
    #     decoded_outputs = self.tokenizer.batch_decode(
    #         torch.cat([target_input_ids[0:1], generated_suffix], dim=1)[:, target_input_ids.shape[-1]:],
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=True,
    #     )

    #     if return_likelihood_ratios:
    #         return decoded_outputs, [float(sum(per_step_log_ratios))]
    #     return decoded_outputs

