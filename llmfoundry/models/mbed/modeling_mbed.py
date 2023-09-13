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
        
        self.temperature = resolved_om_model_config.get('temperature', 1)

        if not pretrained_model_name:
            pretrained_model_name = 'bert-base-uncased'

        config = BertConfig.from_pretrained(pretrained_model_name,
                                            **resolved_om_model_config)
        
        model = BertModel(config, add_pooling_layer=False)
        
        super().__init__(model=model,
                         tokenizer=tokenizer,
                         metrics=[],
                         use_logits=True)

    def forward(self, batch):
        scores, labels = self._compute_scores(batch)
        #self.labels = labels # added for eval, doesn't work
        
        loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(scores, labels)
        
        # Based on https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/simlm/src/models/biencoder_model.py#L60C62-L60C62
        # We are scaling the loss by the world size because we think it will be divided by the world size in the backward pass
        # This is a hacky way of getting around implementing our own backward pass
        # loss *= dist.get_world_size()
        
        return {
            'loss': loss,
            'logits': scores, # This doesn't seem right, but needs to be here for torchmetrics
            'labels': labels
        }

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module: nn.Module):
        return isinstance(module, BertLayer)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module: nn.Module):
        return isinstance(module, BertLayer)
    
    def format_queries_batch(self, batch):
        queries = {}
        for key in batch.keys():
            queries[key] = batch[key][:,0::2,:].reshape(batch[key].size(0), -1)
        
        return queries
    
    def format_passages_batch(self, batch):
        passages = {}
        for key in batch.keys():
            passages[key] = batch[key][:,1::2,:].reshape(batch[key].size(0), -1)
        
        return passages
        
    def _compute_scores(self, batch) -> Tuple:

        # Run Pairs through the encoder separately in two passes, designated as q (query) and p (passage)
        # [batch_size, sequence_length]
        #
        # the pooled_outputs is [batch_size, hidden_size]
        #
        # Note: at some future point we could use the flag 'token_type_ids' which was used in the original
        # BERT formula to keep track of sentences A and sentences B in the next sentence prediction objective
        # function. For now we split even and odd rows
        queries_batch = self.format_queries_batch(batch)
        passages_batch = self.format_passages_batch(batch)
        
        
        # print(self.tokenizer.decode(queries_batch['input_ids'][0]))
        # print(self.tokenizer.decode(passages_batch['input_ids'][0]))

        (q_encoder_outputs, _) = self.model(
                                        input_ids=queries_batch['input_ids'],
                                        token_type_ids=queries_batch.get('token_type_ids', None),
                                        attention_mask=queries_batch.get('attention_mask', None),
                                        position_ids=queries_batch.get('position_ids', None),
                                        masked_tokens_mask=queries_batch.get('masked_tokens_mask', None),
                                    )

        (p_encoder_outputs, _) = self.model(
                                        input_ids=passages_batch['input_ids'],
                                        token_type_ids=passages_batch.get('token_type_ids', None),
                                        attention_mask=passages_batch.get('attention_mask', None),
                                        position_ids=passages_batch.get('position_ids', None),
                                        masked_tokens_mask=passages_batch.get('masked_tokens_mask', None),
                                    )

        q_last_hidden = q_encoder_outputs.masked_fill(~queries_batch.get('attention_mask', None)[..., None].bool(), 0.0)
        q_pooled_outputs = q_last_hidden.sum(dim=1) / queries_batch.get('attention_mask', None).sum(dim=1)[..., None]
        
        p_last_hidden = p_encoder_outputs.masked_fill(~passages_batch.get('attention_mask', None)[..., None].bool(), 0.0)
        p_pooled_outputs = p_last_hidden.sum(dim=1) / passages_batch.get('attention_mask', None).sum(dim=1)[..., None]
        
        #print('>>p_pooled_outputs shape:',p_pooled_outputs.shape)
        
        q_pooled_outputs = F.normalize(q_pooled_outputs, dim=-1) # Todo: should be configurable when L2 normalizing
        p_pooled_outputs = F.normalize(p_pooled_outputs, dim=-1)

        q_pooled_outputs = q_pooled_outputs.contiguous() # Why do we need to make this contiguous?
        p_pooled_outputs = p_pooled_outputs.contiguous() # Why do we need to make this contiguous?

        # all_q_pooled_outputs = dist_gather_tensor(q_pooled_outputs)
        # all_p_pooled_outputs = dist_gather_tensor(p_pooled_outputs)
        
        all_scores, all_labels = self.full_contrastive_scores_and_labels(queries=all_q_pooled_outputs, 
                                                                         passages=all_p_pooled_outputs)
        
        scale = 1 / self.temperature
        
        all_scores = all_scores * scale
        
        # start = dist.get_global_rank() * q_pooled_outputs.shape[0]
        
        # local_query_indices = torch.arange(start, start + q_pooled_outputs.shape[0], dtype=torch.long).to(q_pooled_outputs.device)
        
        # scores = all_scores.index_select(dim=0, index=local_query_indices)
        # labels = all_labels.index_select(dim=0, index=local_query_indices)s
        scores = all_scores
        labels = all_labels
        #print('>>labels',labels.shape) # should be torch.Size([64])
        return scores, labels

    def full_contrastive_scores_and_labels(self, queries: torch.Tensor, passages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # the labels 
        labels = torch.arange(0, passages.shape[0], dtype=torch.long, device=passages.device)

        # this calculates the inner product between query and passage pairs
        qp = torch.mm(queries, passages.t())

        #print('>> qp shape:', qp.shape)

        return qp, labels