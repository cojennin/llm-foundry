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
from torch import nn
from transformers import PreTrainedModel
from composer.metrics.nlp import (LanguageCrossEntropy)
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
        super().__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=True)
        

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
        
        model = BertModel(config, add_pooling_layer=True)
        
        loss_fn_config = om_model_config.get('loss_fn', 'fused_crossentropy')
        if loss_fn_config == 'fused_crossentropy':
            try:
                from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss  # type: ignore # isort: skip

                if config.verbose > 1:
                    warnings.warn('Using Fused Cross Entropy Loss.')
                self.loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
            except:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                    +
                    'and `pip install .[gpu]` if installing from source or `pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy` '
                    +
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.'
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(
                f'Specified loss_fn={self.loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].'
            )
        
        metrics = [
            LanguageCrossEntropy(ignore_index=-100),
        ]
        
        super().__init__(model=model,
                         tokenizer=tokenizer,
                         use_logits=True,
                         metrics=metrics)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        masked_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:

        (encoder_outputs, pooled_outputs) = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            masked_tokens_mask=masked_tokens_mask,
        )
        
        return pooled_outputs
    
    def loss(self, outputs, batch):
        _, targets = batch
        return self.loss_fn(outputs, targets)