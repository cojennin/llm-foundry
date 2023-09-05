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
import torch.nn as nn
import torch.nn.functional as F

from composer.utils import dist

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

# Inspiration from: https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/simlm/src/models/biencoder_model.py#L103
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
        
        metrics = [
            LanguageCrossEntropy(ignore_index=-100),
        ]
        
        super().__init__(model=model,
                         tokenizer=tokenizer,
                         use_logits=True,
                         metrics=metrics)

    def forward(self, batch):
        scores, labels = self._compute_scores(batch)
        
        loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(scores, labels)
        
        return {
            'loss': loss
        }
        
    def _compute_scores(self, batch) -> Tuple:
        (_, pooled_outputs) = self.model(
            input_ids=batch['input_ids'],
            token_type_ids=batch.get('token_type_ids', None),
            attention_mask=batch.get('attention_mask', None),
            position_ids=batch.get('position_ids', None),
            masked_tokens_mask=batch.get('masked_tokens_mask', None),
        )
        
        pooled_outputs = F.normalize(pooled_outputs, dim=-1) # Todo: should be configurable when L2 normalizing
        
        pooled_outputs = pooled_outputs.contiguous() # Why do we need to make this contiguous?

        all_pooled_outputs = torch.cat(dist.all_gather(pooled_outputs), dim=0)
        
        all_scores, all_labels = self.full_contrastive_scores_and_labels(all_pooled_outputs)
        
        scale = 1 / 0.2 # Todo: should be configurable when L2 normalizing, 0.2 should be a temperature arugment
        
        all_scores = all_scores * scale
        
        start = dist.get_global_rank() * all_pooled_outputs.shape[0]
        
        local_query_indices = torch.arange(start, start + pooled_outputs.shape[0], dtype=torch.long).to(pooled_outputs.device)
        
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        labels = all_labels.index_select(dim=0, index=local_query_indices)

        return scores, labels

    def full_contrastive_scores_and_labels(self, passages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        labels = torch.arange(0, passages.shape[0], dtype=torch.long, device=passages.device)

        qk = torch.mm(passages, passages.t())

        return qk, labels