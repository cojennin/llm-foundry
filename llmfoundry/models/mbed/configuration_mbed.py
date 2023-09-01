# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""A HuggingFace-style model configuration."""

import warnings
from typing import Any, Dict, Optional, Union

from transformers import PretrainedConfig


class BertConfig(PretrainedConfig):
    model_type = 'bert'

    def __init__(
        self,
        alibi_starting_size: int = 512,
        attention_probs_dropout_prob: float = 0.0,
        **kwargs: Any,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in Mosaic BERT
                (otherwise, Flash Attention will be off by default). Defaults to 0.0.
        """
        super().__init__(
            attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.alibi_starting_size = alibi_starting_size
