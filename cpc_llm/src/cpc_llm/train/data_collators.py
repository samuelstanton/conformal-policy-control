"""Vendored data collators removed from trl >= 0.14.

DataCollatorForCompletionOnlyLM was removed from trl in favor of
``completion_only_loss=True`` in ``SFTConfig`` with prompt-completion datasets.
This codebase uses single-text formatting with a response template marker, so
we vendor the original class here to avoid restructuring the data pipeline.

See https://github.com/huggingface/trl/discussions/3826
"""

from dataclasses import dataclass
from typing import Any, List, Union

import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Data collator that masks prompt tokens so loss is only on completions.

    Finds ``response_template`` token IDs in each input and sets labels to -100
    for every token before (and including) the template.  Tokens after the
    template keep their original IDs so the cross-entropy loss is computed only
    on the completion.

    Args:
        response_template: Token IDs (list) or string marking the start of the
            completion.  When a string is passed it is encoded with the
            tokenizer.
        tokenizer: The tokenizer used for encoding.
        mlm: Must be ``False`` (causal LM only).
        ignore_index: Label value for masked positions (default -100).
    """

    response_template: Union[str, List[int]] = None
    ignore_index: int = -100

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.response_template, str):
            self.response_template_ids: List[int] = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            self.response_template_ids = list(self.response_template)
        if not self.response_template_ids:
            raise ValueError("response_template must encode to at least one token")

    def torch_call(self, examples: List[Union[List[int], Any, dict]]) -> dict:
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            response_start = self._find_response_start(batch["input_ids"][i])
            if response_start is not None:
                batch["labels"][i, :response_start] = self.ignore_index
            else:
                # Template not found — mask entire sequence so it contributes
                # no loss rather than training on the prompt.
                batch["labels"][i, :] = self.ignore_index
        return batch

    def _find_response_start(self, input_ids: torch.Tensor) -> int | None:
        """Return the index of the first token *after* the response template."""
        template_len = len(self.response_template_ids)
        for idx in range(len(input_ids) - template_len + 1):
            if (
                input_ids[idx : idx + template_len].tolist()
                == self.response_template_ids
            ):
                return idx + template_len
        return None
