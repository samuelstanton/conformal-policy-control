import gc
import logging
import numpy as np
import os
import s3fs
import tempfile
import torch
import torch.distributed.checkpoint as dist_cp

from finetune_utils import maybe_log
from string import Template
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional, Tuple, Union


class ModelClient:
    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        hf_model_name: str = None,  # The name of the model in the HF hub. Only necessary
        # if loading from a sharded FSDP checkpoint.
        logger: logging.Logger = None,
        max_generate_length: int = 500,
        device: Optional[str] = None,
        **model_init_args,
    ):
        self.model_name_or_path = model_name_or_path
        self.hf_model_name = hf_model_name
        self.logger = logger

        self.max_generate_length = max_generate_length

        self.device = device
        self.model_init_args = model_init_args

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
                extra_model_kwargs["device_map"] = "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, trust_remote_code=True, **extra_model_kwargs
            )
            if self.device is not None:
                self.model = self.model.to(self.device)
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
            self.model = AutoModelForCausalLM.from_pretrained(
                td, **self.model_init_args
            )
            if self.device is not None:
                self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(td)
        gc.collect()
        torch.cuda.empty_cache()

    def _convert_checkpoint(
        self, hf_model_name: str, fsdp_model_path: str, output_path: str
    ):
        """
        hf_model_name: Name of model in HF Hub, e.g. "gpt2".
        fsdp_model_path: path to the fsdp checkpoint, for example `/x/checkpoint-xxx/pytorch_model_fsdp_x`
        output_path: output path to save the converted checkpoint
        """
        config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, **self.model_init_args
        ).cuda()
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
    def generate_texts_batched_contrastive(
        self,
        model_b: ModelClient,  # Reference/contrast model
        input_texts: Optional[List[str]] = None,
        batch_size: int = 8,
        alpha: float = 1.0,  # Weight for the ratio term
        beta: float = 0.0,   # Weight for reference model likelihood
        temperature: float = 1.0,
        max_new_tokens: int = 100,
        return_likelihood_ratios: bool = False,
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Generate sequences using contrastive decoding between two models.
        
        Args:
            model_b: Reference model (ModelClient instance)
            input_texts: List of input prompts
            batch_size: Batch size for generation
            alpha: Weight for log(P_A / P_B) term
            beta: Weight for log(P_B) term (negative values encourage diversity)
            temperature: Temperature for sampling
            max_new_tokens: Maximum tokens to generate
            return_likelihood_ratios: If True, return log likelihood ratios for each sequence
            **kwargs: Additional generation arguments
        
        Returns:
            If return_likelihood_ratios=False: List of generated sequences
            If return_likelihood_ratios=True: Tuple of (generated_sequences, log_likelihood_ratios)
        """
        if input_texts is None:
            raise ValueError("input_texts cannot be None for contrastive generation")
        
        output_strs = []
        all_log_likelihood_ratios = []
        
        for batch_start_idx in tqdm(
            range(0, len(input_texts), batch_size), 
            desc="Contrastive generation..."
        ):
            batch_end_idx = batch_start_idx + batch_size
            batch_texts = input_texts[batch_start_idx:batch_end_idx]
            
            # Tokenize input
            input_ids = self._tokenize_batch(batch_texts).input_ids
            if self.device is not None:
                input_ids = input_ids.to(self.device)
            
            # Generate token by token using contrastive decoding
            generated_sequences = input_ids.clone()
            batch_log_likelihood_ratios = []
            
            for step in range(max_new_tokens):
                # Get logits from both models
                outputs_a = self.model(generated_sequences)
                outputs_b = model_b.model(generated_sequences)
                
                logits_a = outputs_a.logits[:, -1, :]  # Last token logits
                logits_b = outputs_b.logits[:, -1, :]  # Last token logits
                
                # Compute log probabilities
                log_probs_a = torch.nn.functional.log_softmax(logits_a / temperature, dim=-1)
                log_probs_b = torch.nn.functional.log_softmax(logits_b / temperature, dim=-1)
                
                # Contrastive score: alpha * log(P_A / P_B) + beta * log(P_B)
                contrastive_scores = alpha * (log_probs_a - log_probs_b) + beta * log_probs_b
                
                # Sample next token
                if temperature > 0:
                    probs = torch.softmax(contrastive_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(contrastive_scores, dim=-1, keepdim=True)
                
                # Store log likelihood ratios for this step if requested
                if return_likelihood_ratios:
                    # Get the log likelihood ratio for the selected tokens
                    selected_log_probs_a = torch.gather(log_probs_a, -1, next_tokens).squeeze(-1)
                    selected_log_probs_b = torch.gather(log_probs_b, -1, next_tokens).squeeze(-1)
                    step_log_ratios = selected_log_probs_a - selected_log_probs_b
                    batch_log_likelihood_ratios.append(step_log_ratios)
                
                # Append to sequences
                generated_sequences = torch.cat([generated_sequences, next_tokens], dim=-1)
                
                # Check for EOS tokens
                if (next_tokens == self.tokenizer.eos_token_id).all():
                    break
            
            # Decode generated sequences (remove input context)
            decoded_outputs = self.tokenizer.batch_decode(
                generated_sequences[:, input_ids.shape[-1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            output_strs.extend(decoded_outputs)
            
            # Aggregate log likelihood ratios for this batch if requested
            if return_likelihood_ratios:
                if batch_log_likelihood_ratios:
                    # Stack all step ratios and sum across tokens for each sequence
                    batch_ratios = torch.stack(batch_log_likelihood_ratios, dim=1)  # [batch_size, num_tokens]
                    sequence_total_ratios = batch_ratios.sum(dim=1)  # Sum across tokens
                    all_log_likelihood_ratios.extend(sequence_total_ratios.cpu().tolist())
                else:
                    # No tokens generated, add zeros
                    all_log_likelihood_ratios.extend([0.0] * len(decoded_outputs))
        
        if return_likelihood_ratios:
            return output_strs, all_log_likelihood_ratios
        else:
            return output_strs




#### Relative to mixture version
# @torch.no_grad()
# def generate_texts_batched_contrastive(
#     self,
#     reference_models: Union[ModelClient, List[ModelClient]],  # Single model or list of reference models
#     input_texts: Optional[List[str]] = None,
#     batch_size: int = 8,
#     alpha: float = 1.0,  # Weight for the ratio term
#     beta: float = 0.0,   # Weight for reference model likelihood
#     temperature: float = 1.0,
#     max_new_tokens: int = 100,
#     return_likelihood_ratios: bool = False,
#     mixture_weights: Optional[List[float]] = None,  # Optional custom weights for mixture
#     **kwargs,
# ) -> Union[List[str], Tuple[List[str], List[float]]]:
#     """
#     Generate sequences using contrastive decoding between target model and mixture of reference models.
    
#     Args:
#         reference_models: Single ModelClient or list of ModelClient instances for mixture
#         input_texts: List of input prompts
#         batch_size: Batch size for generation
#         alpha: Weight for log(P_A / P_mixture) term
#         beta: Weight for log(P_mixture) term (negative values encourage diversity)
#         temperature: Temperature for sampling
#         max_new_tokens: Maximum tokens to generate
#         return_likelihood_ratios: If True, return log likelihood ratios for each sequence
#         mixture_weights: Optional list of weights for each reference model (default: uniform)
#         **kwargs: Additional generation arguments
    
#     Returns:
#         If return_likelihood_ratios=False: List of generated sequences
#         If return_likelihood_ratios=True: Tuple of (generated_sequences, log_likelihood_ratios)
#     """
#     if input_texts is None:
#         raise ValueError("input_texts cannot be None for contrastive generation")
    
#     # Handle single model vs list of models
#     if isinstance(reference_models, ModelClient):
#         reference_models = [reference_models]
    
#     num_ref_models = len(reference_models)
#     if num_ref_models == 0:
#         raise ValueError("At least one reference model must be provided")
    
#     # Set up mixture weights (uniform by default)
#     if mixture_weights is None:
#         mixture_weights = [1.0 / num_ref_models] * num_ref_models
#     else:
#         if len(mixture_weights) != num_ref_models:
#             raise ValueError(f"mixture_weights length ({len(mixture_weights)}) must match number of reference models ({num_ref_models})")
#         # Normalize weights to sum to 1
#         total_weight = sum(mixture_weights)
#         mixture_weights = [w / total_weight for w in mixture_weights]
    
#     output_strs = []
#     all_log_likelihood_ratios = []
    
#     for batch_start_idx in tqdm(
#         range(0, len(input_texts), batch_size), 
#         desc="Contrastive generation with mixture..."
#     ):
#         batch_end_idx = batch_start_idx + batch_size
#         batch_texts = input_texts[batch_start_idx:batch_end_idx]
        
#         # Tokenize input
#         input_ids = self._tokenize_batch(batch_texts).input_ids
#         if self.device is not None:
#             input_ids = input_ids.to(self.device)
        
#         # Generate token by token using contrastive decoding
#         generated_sequences = input_ids.clone()
#         batch_log_likelihood_ratios = []
        
#         for step in range(max_new_tokens):
#             # Get logits from target model
#             outputs_a = self.model(generated_sequences)
#             logits_a = outputs_a.logits[:, -1, :]  # Last token logits
            
#             # Get logits from all reference models
#             ref_logits = []
#             for ref_model in reference_models:
#                 outputs_ref = ref_model.model(generated_sequences)
#                 ref_logits.append(outputs_ref.logits[:, -1, :])
            
#             # Compute log probabilities
#             log_probs_a = torch.nn.functional.log_softmax(logits_a / temperature, dim=-1)
            
#             # Compute mixture log probabilities
#             # P_mixture = sum(w_i * P_i) where w_i are mixture weights
#             # log(P_mixture) = logsumexp(log(w_i) + log(P_i))
#             mixture_log_probs = None
#             for i, (ref_logit, weight) in enumerate(zip(ref_logits, mixture_weights)):
#                 ref_log_probs = torch.nn.functional.log_softmax(ref_logit / temperature, dim=-1)
#                 weighted_log_probs = torch.log(torch.tensor(weight, device=ref_log_probs.device)) + ref_log_probs
                
#                 if mixture_log_probs is None:
#                     mixture_log_probs = weighted_log_probs
#                 else:
#                     mixture_log_probs = torch.logsumexp(
#                         torch.stack([mixture_log_probs, weighted_log_probs]), dim=0
#                     )
            
#             # Contrastive score: alpha * log(P_A / P_mixture) + beta * log(P_mixture)
#             contrastive_scores = alpha * (log_probs_a - mixture_log_probs) + beta * mixture_log_probs
            
#             # Sample next token
#             if temperature > 0:
#                 probs = torch.softmax(contrastive_scores, dim=-1)
#                 next_tokens = torch.multinomial(probs, num_samples=1)
#             else:
#                 # Greedy decoding
#                 next_tokens = torch.argmax(contrastive_scores, dim=-1, keepdim=True)
            
#             # Store log likelihood ratios for this step if requested
#             if return_likelihood_ratios:
#                 # Get the log likelihood ratio for the selected tokens
#                 selected_log_probs_a = torch.gather(log_probs_a, -1, next_tokens).squeeze(-1)
#                 selected_mixture_log_probs = torch.gather(mixture_log_probs, -1, next_tokens).squeeze(-1)
#                 step_log_ratios = selected_log_probs_a - selected_mixture_log_probs
#                 batch_log_likelihood_ratios.append(step_log_ratios)
            
#             # Append to sequences
#             generated_sequences = torch.cat([generated_sequences, next_tokens], dim=-1)
            
#             # Check for EOS tokens
#             if (next_tokens == self.tokenizer.eos_token_id).all():
#                 break
        
#         # Decode generated sequences (remove input context)
#         decoded_outputs = self.tokenizer.batch_decode(
#             generated_sequences[:, input_ids.shape[-1]:],
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True,
#         )
#         output_strs.extend(decoded_outputs)
        
#         # Aggregate log likelihood ratios for this batch if requested
#         if return_likelihood_ratios:
#             if batch_log_likelihood_ratios:
#                 # Stack all step ratios and sum across tokens for each sequence
#                 batch_ratios = torch.stack(batch_log_likelihood_ratios, dim=1)  # [batch_size, num_tokens]
#                 sequence_total_ratios = batch_ratios.sum(dim=1)  # Sum across tokens
#                 all_log_likelihood_ratios.extend(sequence_total_ratios.cpu().tolist())
#             else:
#                 # No tokens generated, add zeros
#                 all_log_likelihood_ratios.extend([0.0] * len(decoded_outputs))
    
#     if return_likelihood_ratios:
#         return output_strs, all_log_likelihood_ratios
#     else:
#         return output_strs



# ### Version that's relative to mixture and also reference models' likelihoods are averaged over their corresponding prompts

# @torch.no_grad()
# def generate_texts_batched_contrastive_mixture(
#     self,
#     reference_models: List[ModelClient],  # List of reference models
#     input_texts_list: List[List[str]],  # List of input prompt lists for each model
#     batch_size: int = 8,
#     alpha: float = 1.0,  # Weight for the ratio term
#     beta: float = 0.0,   # Weight for reference model likelihood
#     temperature: float = 1.0,
#     max_new_tokens: int = 100,
#     return_likelihood_ratios: bool = False,
#     mixture_weights: Optional[List[float]] = None,  # Optional custom weights for mixture
#     **kwargs,
# ) -> Union[List[str], Tuple[List[str], List[float]]]:
#     """
#     Generate sequences using contrastive decoding between target model and mixture of reference models.
#     Each model can have different input prompts.
    
#     Args:
#         reference_models: List of ModelClient instances for mixture
#         input_texts_list: List of input prompt lists, one for each reference model + target model
#                          Length should be len(reference_models) + 1
#                          input_texts_list[i] contains prompts for reference_models[i]
#                          input_texts_list[-1] contains prompts for target model (self)
#         batch_size: Batch size for generation
#         alpha: Weight for log(P_A / P_mixture) term
#         beta: Weight for log(P_mixture) term (negative values encourage diversity)
#         temperature: Temperature for sampling
#         max_new_tokens: Maximum tokens to generate
#         return_likelihood_ratios: If True, return log likelihood ratios for each sequence
#         mixture_weights: Optional list of weights for each reference model (default: uniform)
#         **kwargs: Additional generation arguments
    
#     Returns:
#         If return_likelihood_ratios=False: List of generated sequences
#         If return_likelihood_ratios=True: Tuple of (generated_sequences, log_likelihood_ratios)
#     """
#     num_ref_models = len(reference_models)
#     if num_ref_models == 0:
#         raise ValueError("At least one reference model must be provided")
    
#     # Validate input_texts_list length
#     expected_length = num_ref_models + 1  # +1 for target model
#     if len(input_texts_list) != expected_length:
#         raise ValueError(f"input_texts_list length ({len(input_texts_list)}) must be {expected_length} (num_ref_models + 1)")
    
#     # Validate that all input_texts_list elements have the same length
#     target_input_texts = input_texts_list[-1]  # Target model prompts
#     num_sequences = len(target_input_texts)
    
#     for i, ref_input_texts in enumerate(input_texts_list[:-1]):
#         if len(ref_input_texts) != num_sequences:
#             raise ValueError(f"input_texts_list[{i}] length ({len(ref_input_texts)}) must match target model prompts length ({num_sequences})")
    
#     # Set up mixture weights (uniform by default)
#     if mixture_weights is None:
#         mixture_weights = [1.0 / num_ref_models] * num_ref_models
#     else:
#         if len(mixture_weights) != num_ref_models:
#             raise ValueError(f"mixture_weights length ({len(mixture_weights)}) must match number of reference models ({num_ref_models})")
#         # Normalize weights to sum to 1
#         total_weight = sum(mixture_weights)
#         mixture_weights = [w / total_weight for w in mixture_weights]
    
#     output_strs = []
#     all_log_likelihood_ratios = []


#     ## Get batch prompts for reference models
#     ref_input_texts_list = input_texts_list[:-1]


#     # Tokenize input for each reference model
#     ref_input_ids_list = []
#     for ref_input_texts in ref_input_texts_list:
#         ref_input_ids = self._tokenize_batch(ref_input_texts).input_ids
#         if self.device is not None:
#             ref_input_ids = ref_input_ids.to(self.device)
#         ref_input_ids_list.append(ref_input_ids)

    
#     for batch_start_idx in tqdm(
#         range(0, num_sequences, batch_size), 
#         desc="Contrastive generation with mixture and different prompts..."
#     ):
#         batch_end_idx = batch_start_idx + batch_size
        
#         # Get batch prompts for target model
#         target_batch_texts = target_input_texts[batch_start_idx:batch_end_idx]

        
#         # Tokenize input for target model
#         target_input_ids = self._tokenize_batch(target_batch_texts).input_ids
#         if self.device is not None:
#             target_input_ids = target_input_ids.to(self.device)
        
        
#         # Generate token by token using contrastive decoding
#         generated_sequences = target_input_ids.clone()
#         batch_log_likelihood_ratios = []
        
#         for step in range(max_new_tokens):
#             # Get logits from target model
#             outputs_a = self.model(generated_sequences)
#             logits_a = outputs_a.logits[:, -1, :]  # Last token logits
            
#             # Get logits from all reference models (each with their own prompts)
#             ref_logits = []
#             for i, (ref_model, ref_input_ids) in enumerate(zip(reference_models, ref_input_ids_list)):
#                 # For reference models, we need to use their specific input prompts
#                 # and append the generated tokens so far
#                 ref_sequences = torch.cat([ref_input_ids, generated_sequences[:, target_input_ids.shape[-1]:]], dim=-1)
#                 outputs_ref = ref_model.model(ref_sequences)
#                 ref_logits.append(outputs_ref.logits[:, -1, :])
            
#             # Compute log probabilities
#             log_probs_a = torch.nn.functional.log_softmax(logits_a / temperature, dim=-1)
            
#             # Compute mixture log probabilities
#             # P_mixture = sum(w_i * P_i) where w_i are mixture weights
#             # log(P_mixture) = logsumexp(log(w_i) + log(P_i))
#             mixture_log_probs = None
#             for i, (ref_logit, weight) in enumerate(zip(ref_logits, mixture_weights)):
#                 ref_log_probs = torch.nn.functional.log_softmax(ref_logit / temperature, dim=-1)
#                 weighted_log_probs = torch.log(torch.tensor(weight, device=ref_log_probs.device)) + ref_log_probs
                
#                 if mixture_log_probs is None:
#                     mixture_log_probs = weighted_log_probs
#                 else:
#                     mixture_log_probs = torch.logsumexp(
#                         torch.stack([mixture_log_probs, weighted_log_probs]), dim=0
#                     )
            
#             # Contrastive score: alpha * log(P_A / P_mixture) + beta * log(P_mixture)
#             contrastive_scores = alpha * (log_probs_a - mixture_log_probs) + beta * mixture_log_probs
            
#             # Sample next token
#             if temperature > 0:
#                 probs = torch.softmax(contrastive_scores, dim=-1)
#                 next_tokens = torch.multinomial(probs, num_samples=1)
#             else:
#                 # Greedy decoding
#                 next_tokens = torch.argmax(contrastive_scores, dim=-1, keepdim=True)
            
#             # Store log likelihood ratios for this step if requested
#             if return_likelihood_ratios:
#                 # Get the log likelihood ratio for the selected tokens
#                 selected_log_probs_a = torch.gather(log_probs_a, -1, next_tokens).squeeze(-1)
#                 selected_mixture_log_probs = torch.gather(mixture_log_probs, -1, next_tokens).squeeze(-1)
#                 step_log_ratios = selected_log_probs_a - selected_mixture_log_probs
#                 batch_log_likelihood_ratios.append(step_log_ratios)
            
#             # Append to sequences
#             generated_sequences = torch.cat([generated_sequences, next_tokens], dim=-1)
            
#             # Check for EOS tokens
#             if (next_tokens == self.tokenizer.eos_token_id).all():
#                 break
        
#         # Decode generated sequences (remove input context)
#         decoded_outputs = self.tokenizer.batch_decode(
#             generated_sequences[:, target_input_ids.shape[-1]:],
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True,
#         )
#         output_strs.extend(decoded_outputs)
        
#         # Aggregate log likelihood ratios for this batch if requested
#         if return_likelihood_ratios:
#             if batch_log_likelihood_ratios:
#                 # Stack all step ratios and sum across tokens for each sequence
#                 batch_ratios = torch.stack(batch_log_likelihood_ratios, dim=1)  # [batch_size, num_tokens]
#                 sequence_total_ratios = batch_ratios.sum(dim=1)  # Sum across tokens
#                 all_log_likelihood_ratios.extend(sequence_total_ratios.cpu().tolist())
#             else:
#                 # No tokens generated, add zeros
#                 all_log_likelihood_ratios.extend([0.0] * len(decoded_outputs))
    
#     if return_likelihood_ratios:
#         return output_strs, all_log_likelihood_ratios
#     else:
#         return output_strs




    # @torch.no_grad()
    # def generate_texts_batched_contrastive_mixture(
    #     self,
    #     reference_models: List[ModelClient],  # List of reference models
    #     input_texts_list: List[List[str]],  # List of input prompt lists for each model
    #     batch_size: int = 8,
    #     alpha: float = 1.0,  # Weight for the ratio term
    #     beta: float = 0.0,   # Weight for reference model likelihood
    #     temperature: float = 1.0,
    #     max_new_tokens: int = 100,
    #     return_likelihood_ratios: bool = False,
    #     mixture_weights: Optional[List[float]] = None,  # Optional custom weights for mixture
    #     **kwargs,
    # ) -> Union[List[str], Tuple[List[str], List[float]]]:
    #     """
    #     Generate sequences using contrastive decoding between target model and mixture of reference models.
    #     Each model can have different input prompts, and reference model likelihoods are averaged over their prompts.
        
    #     Args:
    #         reference_models: List of ModelClient instances for mixture
    #         input_texts_list: List of input prompt lists, one for each reference model + target model
    #                         Length should be len(reference_models) + 1
    #                         input_texts_list[i] contains prompts for reference_models[i]
    #                         input_texts_list[-1] contains prompts for target model (self)
    #         batch_size: Batch size for generation
    #         alpha: Weight for log(P_A / P_mixture) term
    #         beta: Weight for log(P_mixture) term (negative values encourage diversity)
    #         temperature: Temperature for sampling
    #         max_new_tokens: Maximum tokens to generate
    #         return_likelihood_ratios: If True, return log likelihood ratios for each sequence
    #         mixture_weights: Optional list of weights for each reference model (default: uniform)
    #         **kwargs: Additional generation arguments
        
    #     Returns:
    #         If return_likelihood_ratios=False: List of generated sequences
    #         If return_likelihood_ratios=True: Tuple of (generated_sequences, log_likelihood_ratios)
    #     """
    #     num_ref_models = len(reference_models)
    #     if num_ref_models == 0:
    #         raise ValueError("At least one reference model must be provided")
        
    #     # Validate input_texts_list length
    #     expected_length = num_ref_models + 1  # +1 for target model
    #     if len(input_texts_list) != expected_length:
    #         raise ValueError(f"input_texts_list length ({len(input_texts_list)}) must be {expected_length} (num_ref_models + 1)")
        
    #     ## Drew: Ideally shouldn't need to have these all be the same length
    #     # Validate that all input_texts_list elements have the same length
    #     target_input_texts = input_texts_list[-1]  # Target model prompts
    #     num_sequences = len(target_input_texts)
        
    #     for i, ref_input_texts in enumerate(input_texts_list[:-1]):
    #         if len(ref_input_texts) != num_sequences:
    #             raise ValueError(f"input_texts_list[{i}] length ({len(ref_input_texts)}) must match target model prompts length ({num_sequences})")
        
    #     ## Drew: Mixture weights should be equal to number of calibration data for each of those models
    #     # Set up mixture weights (uniform by default)
    #     if mixture_weights is None:
    #         mixture_weights = [1.0 / num_ref_models] * num_ref_models
    #     else:
    #         if len(mixture_weights) != num_ref_models:
    #             raise ValueError(f"mixture_weights length ({len(mixture_weights)}) must match number of reference models ({num_ref_models})")
    #         # Normalize weights to sum to 1
    #         total_weight = sum(mixture_weights)
    #         mixture_weights = [w / total_weight for w in mixture_weights]
        
    #     # Tokenize input for all reference models (outside batch loop)
    #     ref_input_texts_list = input_texts_list[:-1]
    #     ref_input_ids_list = []
    #     for ref_input_texts in ref_input_texts_list:
    #         ref_input_ids = self._tokenize_batch(ref_input_texts).input_ids
    #         if self.device is not None:
    #             ref_input_ids = ref_input_ids.to(self.device)
    #         ref_input_ids_list.append(ref_input_ids)
        
    #     output_strs = []
    #     all_log_likelihood_ratios = []
        
    #     # for batch_start_idx in tqdm(
    #     #     range(0, num_sequences, batch_size), 
    #     #     desc="Contrastive generation with target over mixture of reference policies..."
    #     # ):
    #     #     batch_end_idx = batch_start_idx + batch_size
            
    #     # Get batch prompts for target model only
    #     # target_batch_texts = target_input_texts[batch_start_idx:batch_end_idx]
        
    #     # Tokenize input for target model
    #     target_input_ids = self._tokenize_batch(target_input_texts).input_ids
    #     if self.device is not None:
    #         target_input_ids = target_input_ids.to(self.device)
        
    #     # Generate token by token using contrastive decoding
    #     generated_sequences = target_input_ids.clone()
    #     batch_log_likelihood_ratios = []
        
    #     for step in range(max_new_tokens):
    #         # Get logits from target model
    #         outputs_a = self.model(generated_sequences)
    #         logits_a = outputs_a.logits[:, -1, :]  # Last token logits
            
    #         # Get logits from all reference models (averaged over their prompts)
    #         ref_logits = []
    #         for i, ref_model in enumerate(reference_models):
    #             ref_input_ids = ref_input_ids_list[i]
                
    #             # Compute logits for each prompt in this reference model's input set
    #             ref_logits_j = []
    #             for j in range(ref_input_ids.shape[0]):  # Iterate over all prompts for this ref model
    #                 # Construct sequence: ref prompt + generated tokens so far
    #                 ref_sequence = torch.cat([ref_input_ids[j:j+1], generated_sequences[:, target_input_ids.shape[-1]:]], dim=-1)
    #                 outputs_ref = ref_model.model(ref_sequence)
    #                 ref_logits_j.append(outputs_ref.logits[:, -1, :])  # Last token logits
                
    #             # Average logits over all prompts for this reference model
    #             ref_logits_j = torch.stack(ref_logits_j, dim=0)  # [num_prompts, batch_size, vocab_size]
    #             ref_logits.append(ref_logits_j.mean(dim=0))  # Average over prompts: [batch_size, vocab_size]
            
    #         # Compute log probabilities

    #         log_probs_a = torch.nn.functional.log_softmax(logits_a / temperature, dim=-1)
            
    #         # Compute mixture log probabilities
    #         # P_mixture = sum(w_i * P_i) where w_i are mixture weights
    #         # log(P_mixture) = logsumexp(log(w_i) + log(P_i))
    #         mixture_log_probs = None
    #         for i, (ref_logit, weight) in enumerate(zip(ref_logits, mixture_weights)):
    #             ref_log_probs = torch.nn.functional.log_softmax(ref_logit / temperature, dim=-1)
    #             weighted_log_probs = torch.log(torch.tensor(weight, device=ref_log_probs.device)) + ref_log_probs
                
    #             if mixture_log_probs is None:
    #                 mixture_log_probs = weighted_log_probs
    #             else:
    #                 mixture_log_probs = torch.logsumexp(
    #                     torch.stack([mixture_log_probs, weighted_log_probs]), dim=0
    #                 )
            
    #         # Contrastive score: alpha * log(P_A / P_mixture) + beta * log(P_mixture)
    #         contrastive_scores = alpha * (log_probs_a - mixture_log_probs) + beta * mixture_log_probs
            
    #         # Sample next token
    #         if temperature > 0:
    #             probs = torch.softmax(contrastive_scores, dim=-1)
    #             next_tokens = torch.multinomial(probs, num_samples=1)
    #         else:
    #             # Greedy decoding
    #             next_tokens = torch.argmax(contrastive_scores, dim=-1, keepdim=True)
            
    #         # Store log likelihood ratios for this step if requested
    #         if return_likelihood_ratios:
    #             # Get the log likelihood ratio for the selected tokens
    #             selected_log_probs_a = torch.gather(log_probs_a, -1, next_tokens).squeeze(-1)
    #             selected_mixture_log_probs = torch.gather(mixture_log_probs, -1, next_tokens).squeeze(-1)
    #             step_log_ratios = selected_log_probs_a - selected_mixture_log_probs
    #             batch_log_likelihood_ratios.append(step_log_ratios)
            
    #         # Append to sequences
    #         generated_sequences = torch.cat([generated_sequences, next_tokens], dim=-1)
            
    #         # Check for EOS tokens
    #         if (next_tokens == self.tokenizer.eos_token_id).all():
    #             break
        
    #     # Decode generated sequences (remove input context)
    #     decoded_outputs = self.tokenizer.batch_decode(
    #         generated_sequences[:, target_input_ids.shape[-1]:],
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=True,
    #     )
    #     output_strs.extend(decoded_outputs)
        
    #     # Aggregate log likelihood ratios for this batch if requested
    #     if return_likelihood_ratios:
    #         if batch_log_likelihood_ratios:
    #             # Stack all step ratios and sum across tokens for each sequence
    #             batch_ratios = torch.stack(batch_log_likelihood_ratios, dim=1)  # [batch_size, num_tokens]
    #             sequence_total_ratios = batch_ratios.sum(dim=1)  # Sum across tokens
    #             all_log_likelihood_ratios.extend(sequence_total_ratios.cpu().tolist())
    #         else:
    #             # No tokens generated, add zeros
    #             all_log_likelihood_ratios.extend([0.0] * len(decoded_outputs))
        
    #     if return_likelihood_ratios:
    #         return output_strs, all_log_likelihood_ratios
    #     else:
    #         return output_strs


    @torch.no_grad()
    def generate_texts_batched_contrastive(
        self,
        reference_models: List[ModelClient],  # List of reference models
        input_texts_list: List[List[str]],  # List of input prompt lists for each model
        batch_size: int = 8,  # Not used anymore, kept for compatibility
        alpha: float = 1.0,  # Weight for the ratio term
        beta: float = 0.0,   # Weight for reference model likelihood
        temperature: float = 1.0,
        max_new_tokens: int = 100,
        return_likelihood_ratios: bool = False,
        mixture_weights: Optional[List[float]] = None,  # Optional custom weights for mixture
        **kwargs,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Generate sequences using contrastive decoding between target model and mixture of reference models.
        Each model's likelihoods are averaged over their corresponding input prompts.
        
        Args:
            reference_models: List of ModelClient instances for mixture
            input_texts_list: List of input prompt lists, one for each reference model + target model
                            Length should be len(reference_models) + 1
                            input_texts_list[i] contains prompts for reference_models[i]
                            input_texts_list[-1] contains prompts for target model (self)
            batch_size: Not used anymore, kept for compatibility
            alpha: Weight for log(P_A / P_mixture) term
            beta: Weight for log(P_mixture) term (negative values encourage diversity)
            temperature: Temperature for sampling
            max_new_tokens: Maximum tokens to generate
            return_likelihood_ratios: If True, return log likelihood ratios for each sequence
            mixture_weights: Optional list of weights for each reference model (default: uniform)
            **kwargs: Additional generation arguments
        
        Returns:
            If return_likelihood_ratios=False: List of generated sequences
            If return_likelihood_ratios=True: Tuple of (generated_sequences, log_likelihood_ratios)
        """
        num_ref_models = len(reference_models)
        if num_ref_models == 0:
            raise ValueError("At least one reference model must be provided")
        
        # Validate input_texts_list length
        expected_length = num_ref_models + 1  # +1 for target model
        if len(input_texts_list) != expected_length:
            raise ValueError(f"input_texts_list length ({len(input_texts_list)}) must be {expected_length} (num_ref_models + 1)")
        
        # Validate that all input_texts_list elements have the same length
        target_input_texts = input_texts_list[-1]  # Target model prompts
        num_sequences = len(target_input_texts)
        
        for i, ref_input_texts in enumerate(input_texts_list[:-1]):
            if len(ref_input_texts) != num_sequences:
                raise ValueError(f"input_texts_list[{i}] length ({len(ref_input_texts)}) must match target model prompts length ({num_sequences})")
        
        # Set up mixture weights (uniform by default)
        if mixture_weights is None:
            mixture_weights = [1.0 / num_ref_models] * num_ref_models
        else:
            if len(mixture_weights) != num_ref_models:
                raise ValueError(f"mixture_weights length ({len(mixture_weights)}) must match number of reference models ({num_ref_models})")
            # Normalize weights to sum to 1
            total_weight = sum(mixture_weights)
            mixture_weights = [w / total_weight for w in mixture_weights]
        
        # Tokenize input for all models (outside generation loop)
        target_input_ids = self._tokenize_batch(target_input_texts).input_ids
        if self.device is not None:
            target_input_ids = target_input_ids.to(self.device)
        
        ref_input_texts_list = input_texts_list[:-1]
        ref_input_ids_list = []
        for ref_input_texts in ref_input_texts_list:
            ref_input_ids = self._tokenize_batch(ref_input_texts).input_ids
            if self.device is not None:
                ref_input_ids = ref_input_ids.to(self.device)
            ref_input_ids_list.append(ref_input_ids)
        
        # Initialize with first target prompt for generation
        generated_sequences = target_input_ids[0:1].clone()  # Start with first prompt
        all_log_likelihood_ratios = []
        
        for step in tqdm(range(max_new_tokens), desc="Contrastive generation with averaged prompts..."):
            # Get logits from target model (averaged over all target prompts)
            target_logits_j = []
            for j in range(target_input_ids.shape[0]):  # Iterate over all target prompts
                # Construct sequence: target prompt + generated tokens so far
                target_sequence = torch.cat([target_input_ids[j:j+1], generated_sequences[:, target_input_ids.shape[-1]:]], dim=-1)
                outputs_a = self.model(target_sequence)
                target_logits_j.append(outputs_a.logits[:, -1, :])  # Last token logits
            
            # Average target logits over all prompts
            target_logits_j = torch.stack(target_logits_j, dim=0)  # [num_prompts, 1, vocab_size]
            logits_a = target_logits_j.mean(dim=0)  # Average over prompts: [1, vocab_size]
            
            # Get logits from all reference models (averaged over their prompts)
            ref_logits = []
            for i, ref_model in enumerate(reference_models):
                ref_input_ids = ref_input_ids_list[i]
                
                # Compute logits for each prompt in this reference model's input set
                ref_logits_j = []
                for j in range(ref_input_ids.shape[0]):  # Iterate over all prompts for this ref model
                    # Construct sequence: ref prompt + generated tokens so far
                    ref_sequence = torch.cat([ref_input_ids[j:j+1], generated_sequences[:, target_input_ids.shape[-1]:]], dim=-1)
                    outputs_ref = ref_model.model(ref_sequence)
                    ref_logits_j.append(outputs_ref.logits[:, -1, :])  # Last token logits
                
                # Average logits over all prompts for this reference model
                ref_logits_j = torch.stack(ref_logits_j, dim=0)  # [num_prompts, 1, vocab_size]
                ref_logits.append(ref_logits_j.mean(dim=0))  # Average over prompts: [1, vocab_size]
            
            # Compute log probabilities
            log_probs_a = torch.nn.functional.log_softmax(logits_a / temperature, dim=-1)
            
            # Compute mixture log probabilities
            # P_mixture = sum(w_i * P_i) where w_i are mixture weights
            # log(P_mixture) = logsumexp(log(w_i) + log(P_i))
            mixture_log_probs = None
            for i, (ref_logit, weight) in enumerate(zip(ref_logits, mixture_weights)):
                ref_log_probs = torch.nn.functional.log_softmax(ref_logit / temperature, dim=-1)
                weighted_log_probs = torch.log(torch.tensor(weight, device=ref_log_probs.device)) + ref_log_probs
                
                if mixture_log_probs is None:
                    mixture_log_probs = weighted_log_probs
                else:
                    mixture_log_probs = torch.logsumexp(
                        torch.stack([mixture_log_probs, weighted_log_probs]), dim=0
                    )
            
            # Contrastive score: alpha * log(P_A / P_mixture) + beta * log(P_mixture)
            contrastive_scores = alpha * (log_probs_a - mixture_log_probs) + beta * mixture_log_probs
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(contrastive_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(contrastive_scores, dim=-1, keepdim=True)
            
            # Store log likelihood ratios for this step if requested
            if return_likelihood_ratios:
                # Get the log likelihood ratio for the selected tokens
                selected_log_probs_a = torch.gather(log_probs_a, -1, next_tokens).squeeze(-1)
                selected_mixture_log_probs = torch.gather(mixture_log_probs, -1, next_tokens).squeeze(-1)
                step_log_ratios = selected_log_probs_a - selected_mixture_log_probs
                all_log_likelihood_ratios.append(step_log_ratios.item())
            
            # Append to sequences
            generated_sequences = torch.cat([generated_sequences, next_tokens], dim=-1)
            
            # Check for EOS tokens
            if (next_tokens == self.tokenizer.eos_token_id).all():
                break
        
        # Decode generated sequences (remove input context)
        decoded_outputs = self.tokenizer.batch_decode(
            generated_sequences[:, target_input_ids.shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        if return_likelihood_ratios:
            total_log_ratio = sum(all_log_likelihood_ratios)
            return decoded_outputs, [total_log_ratio]
        else:
            return decoded_outputs





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
            probs = torch.exp(torch.nn.functional.log_softmax(scores, dim=-1))
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
            all_likelihoods.extend(avg_likelihoods)
        return all_likelihoods
