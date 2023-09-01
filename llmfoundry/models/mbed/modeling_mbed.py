# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a MosaicBERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nns
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from composer.metrics.nlp import (BinaryF1Score, LanguageCrossEntropy,
                                  MaskedAccuracy)
from composer.models.huggingface import HuggingFaceModel
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from llmfoundry.models.mosaicbert import BertModel
from llmfoundry.models.mosaicbert.configuration_mosaicbert import BertConfig
from llmfoundry.models.mpt.configuration_mpt import MPTConfig

from llmfoundry.models.utils.bert_padding import index_put_first_axis

all = [
    'ComposerMBed'
]

logger = logging.getLogger(__name__)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class MBed(PreTrainedModel):
    
    def __init__(self, config: MPTConfig):
        self.bert = BertModel(config, add_pooling_layer=True)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        masked_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:

        (encoder_outputs, pooled_outputs) = self.bert (
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            masked_tokens_mask=masked_tokens_mask,
        )
        
        return pooled_outputs

class ComposerMBed(HuggingFaceModel):
    """Mosaic MBed model based on |:hugging_face:| Transformers."""

    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[Tokenizer] = None,
    ):
        resolved_om_model_config = om.to_container(om_model_config,
                                                   resolve=True)
        
        pretrained_model_name = resolved_om_model_config.get(
            'pretrained_model_name')

        if not pretrained_model_name:
            pretrained_model_name = 'bert-base-uncased'

        config = BertConfig.from_pretrained(pretrained_model_name,
                                            **resolved_om_model_config)
        
        model = MBed(config, add_pooling_layer=True)
        
        metrics = [
            LanguageCrossEntropy(ignore_index=-100),
        ]
        
        super().__init__(model=model,
                         tokenizer=tokenizer,
                         use_logits=True,
                         metrics=metrics)
        
    