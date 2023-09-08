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
from llmfoundry.models.layers.mosaicbert_layers import BertLayer

all = [
    'ComposerMBed'
]

logger = logging.getLogger(__name__)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# Todo: does this exist somewhere? Move this to a util?
def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(all_tensors, t)

    all_tensors[dist.get_global_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors

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
        
        # metrics = [
        #     LanguageCrossEntropy(ignore_index=-100),
        # ]
        
        super().__init__(model=model,
                         tokenizer=tokenizer)

    def forward(self, batch):
        scores, labels = self._compute_scores(batch)
        
        loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(scores, labels)
        
        return {
            'loss': loss,
            'logits': labels
        }

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module: nn.Module):
        return isinstance(module, BertLayer)
        
    def _compute_scores(self, batch) -> Tuple:

        # Run Pairs through the encoder separately in two passes, designated as q (query) and p (passage)
        # [batch_size, sequence_length]
        #
        # the pooled_outputs is [batch_size, hidden_size]
        #
        # Note: at some future point we could use the flag 'token_type_ids' which was used in the original
        # BERT formula to keep track of sentences A and sentences B in the next sentence prediction objective
        # function. For now we split even and odd rows
        queries = batch['input_ids']#[0::2,:]
        passages = batch['input_ids']#[1::2,:]

        (_, q_pooled_outputs) = self.model(
                                        input_ids=queries,
                                        token_type_ids=batch.get('token_type_ids', None),
                                        attention_mask=batch.get('attention_mask', None),
                                        position_ids=batch.get('position_ids', None),
                                        masked_tokens_mask=batch.get('masked_tokens_mask', None),
                                    )

        (_, p_pooled_outputs) = self.model(
                                        input_ids=passages,
                                        token_type_ids=batch.get('token_type_ids', None),
                                        attention_mask=batch.get('attention_mask', None),
                                        position_ids=batch.get('position_ids', None),
                                        masked_tokens_mask=batch.get('masked_tokens_mask', None),
                                    )

        #print('>>p_pooled_outputs shape:',p_pooled_outputs.shape)
        
        q_pooled_outputs = F.normalize(q_pooled_outputs, dim=-1) # Todo: should be configurable when L2 normalizing
        p_pooled_outputs = F.normalize(p_pooled_outputs, dim=-1)

        q_pooled_outputs = q_pooled_outputs.contiguous() # Why do we need to make this contiguous?
        p_pooled_outputs = p_pooled_outputs.contiguous() # Why do we need to make this contiguous?

        all_q_pooled_outputs = dist_gather_tensor(q_pooled_outputs)
        all_p_pooled_outputs = dist_gather_tensor(p_pooled_outputs)
        
        all_scores, all_labels = self.full_contrastive_scores_and_labels(queries=all_q_pooled_outputs, 
                                                                         passages=all_p_pooled_outputs)
        
        # scale = 1 / 0.2 # Todo: should be configurable when L2 normalizing, 0.2 should be a temperature arugment
        
        # all_scores = all_scores * scale
        
        start = dist.get_global_rank() * all_q_pooled_outputs.shape[0]
        
        local_query_indices = torch.arange(start, start + q_pooled_outputs.shape[0], dtype=torch.long).to(q_pooled_outputs.device)
        
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        labels = all_labels.index_select(dim=0, index=local_query_indices)

        return scores, labels

    def full_contrastive_scores_and_labels(self, queries: torch.Tensor, passages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # the labels 
        labels = torch.arange(0, passages.shape[0], dtype=torch.long, device=passages.device)

        # this calculates the inner product between query and passage pairs
        qp = torch.mm(queries, passages.t())

        #print('>> qp shape:', qp.shape)

        return qp, labels