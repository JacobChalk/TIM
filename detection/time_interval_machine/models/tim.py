import torch.nn.functional as F
import numpy as np
import torch
import math

from torch import nn

import time_interval_machine.models.helpers.head as head
import time_interval_machine.utils.logging as logging

from time_interval_machine.models.helpers.encodings import AudioVisualFeatureEncoding, VisualFeatureEncoding, AudioFeatureEncoding
from time_interval_machine.models.helpers.transformers import TransformerEncoder, TransformerEncoderLayer


logger = logging.get_logger(__name__)

class TIM(nn.Module):
    def __init__(self,
                num_class,
                visual_input_dim=1024,
                audio_input_dim=2304,
                feat_drop=0.5,
                seq_drop=0.5,
                d_model=512,
                feedfoward_scale=4,
                nhead=8,
                num_layers=6,
                enc_dropout=0.1,
                input_modality="audio_visual",
                data_modality="audio_visual",
                num_feats=50,
                include_verb_noun=True,
                iou_threshold=0.25,
                label_smoothing=0.9
            ):
        super(TIM, self).__init__()

        self.input_modality = input_modality
        self.data_modality = data_modality

        self.visual_input_dim = visual_input_dim
        self.audio_input_dim = audio_input_dim
        self.feat_drop=feat_drop
        self.seq_drop = seq_drop

        self.d_model = d_model
        self.dim_feedforward = d_model*feedfoward_scale
        self.nhead = nhead
        self.num_layers = num_layers
        self.enc_dropout = enc_dropout

        self.num_feats = num_feats
        self.num_class = num_class
        self.include_verb_noun = include_verb_noun
        self.iou_threshold = iou_threshold
        self.label_smoothing = label_smoothing

        logger.info("Building {} Transformer with {}-D, {} heads, and {} layers.".format(
                                                             self.input_modality,
                                                             2*self.d_model,
                                                             self.nhead,
                                                             self.num_layers
                                                         )
                                                     )
        self._create_model()

    def _create_model(self):
        self.time_mlp = nn.Sequential(
                            nn.Linear(2, self.d_model),
                            nn.ReLU(),
                            nn.Linear(self.d_model, self.d_model),
                            nn.ReLU(),
                            nn.Linear(self.d_model, self.d_model),
                            nn.ReLU(),
                            nn.LayerNorm(self.d_model)
                        )

        if self.input_modality == "audio_visual":
            self.feature_encoding = AudioVisualFeatureEncoding(
                                        self.visual_input_dim,
                                        self.audio_input_dim,
                                        self.d_model,
                                        self.feat_drop,
                                        self.seq_drop,
                                        self.num_feats,
                                        self.data_modality
                                    )
            self.num_feats *= 2
        elif self.input_modality == "visual":
            self.feature_encoding = VisualFeatureEncoding(
                                        self.visual_input_dim,
                                        self.d_model,
                                        self.feat_drop,
                                        self.seq_drop,
                                        self.num_feats
                                    )

        else:
            self.feature_encoding = AudioFeatureEncoding(
                                        self.audio_input_dim,
                                        self.d_model,
                                        self.feat_drop,
                                        self.seq_drop,
                                        self.num_feats
                                    )


        if self.data_modality == "audio_visual":
            self.cls_head = head.AudioVisualCLSHead(self.num_class, 2*self.d_model)
            self.reg_head = head.AudioVisualRegHead(2*self.d_model)
        elif self.data_modality == "visual":
            self.cls_head = head.VisualCLSHead(self.num_class[0], 2*self.d_model)
            self.reg_head = head.VisualRegHead(2*self.d_model)
        else:
            self.cls_head = head.AudioCLSHead(self.num_class[1], 2*self.d_model)
            self.reg_head = head.AudioRegHead(2*self.d_model)

        encoder_layer = TransformerEncoderLayer(
                            d_model=2*self.d_model,
                            nhead=self.nhead,
                            dim_feedforward=self.dim_feedforward,
                            dropout=self.enc_dropout,
                            activation='gelu'
                        )

        self.backbone = TransformerEncoder(
                                        encoder_layer,
                                        num_layers=self.num_layers
                                    )

        # For MLP
        self.drloc_mlp = nn.Sequential(
                                nn.Linear(4*self.d_model, self.d_model),
                                nn.ReLU(),
                                nn.Linear(self.d_model, self.d_model),
                                nn.ReLU(),
                                nn.Linear(self.d_model, 1)
                            )

        self.train_pool = self.generate_queries(query_size=0.005)
        self.inference_queries = self.generate_queries(query_size=0.01)
        self.num_queries = self.inference_queries.shape[1]

    def generate_queries(self, query_size):
        queries = []
        while query_size < 1.0:
            start_times = torch.arange(0.0, 1.0, step=query_size / 2)
            end_times = start_times + query_size
            layer_times = torch.stack([start_times, end_times], dim=-1)
            layer_times = torch.round(layer_times, decimals=3)
            queries.append(layer_times)
            query_size *= 2

        queries = torch.concat(queries, dim=0).unsqueeze(0)
        return queries
    
    def assign_positive_labels(self, modality, query_labels):
        if modality == "visual":
            verb_labels = torch.empty(size=(0,)).to(device=query_labels.device)
            noun_labels = torch.empty(size=(0,)).to(device=query_labels.device)
            num_actions = self.num_class[0]

            if self.include_verb_noun:
                num_verbs = self.num_class[0][0]
                num_nouns = self.num_class[0][1]
                num_actions = self.num_class[0][2]

                query_labels[:, 0].masked_fill_(query_labels[:, 0]==-1, num_verbs)
                query_labels[:, 1].masked_fill_(query_labels[:, 1]==-1, num_nouns)

                # Smooth labels
                verb_labels = ((F.one_hot(query_labels[:, 0], num_verbs+1) * self.label_smoothing) + ((1 - self.label_smoothing) / (num_verbs+1)))[:, :-1]
                noun_labels = ((F.one_hot(query_labels[:, 1], num_nouns+1) * self.label_smoothing) + ((1 - self.label_smoothing) / (num_nouns+1)))[:, :-1]

            actions_labels = query_labels[:, 2]
            query_labels[:, 2].masked_fill_(query_labels[:, 2]==-1, num_actions)
            actions_labels = ((F.one_hot(query_labels[:, 2], num_actions+1) * self.label_smoothing) + ((1 - self.label_smoothing) / (num_actions+1)))[:, :-1]
            query_labels = [verb_labels, noun_labels, actions_labels]
        else:
            num_actions = self.num_class[1]
            query_labels.masked_fill_(query_labels==-1, num_actions)
            query_labels = ((F.one_hot(query_labels[:, -1], num_actions+1) * self.label_smoothing) + ((1 - self.label_smoothing) / (num_actions+1)))[:, :-1]
        
        return query_labels

    def get_query_ious(self, queries, target_segs):
        """Calculates all ious between queries and target segs

        Args:
            queries (torch.Tensor): Contains the time interval queries for the
            given modality
            target_segs (torch.Tensor): Contains the time interval of targets 
            for the given modality
        Returns:
            ious (torch.Tensor[B, N_q*N_a]): The IOUs between all queries and
            all targets
        """
        query_starts, query_ends = queries[:, :, :, 0], queries[:, :, :, 1]
        gt_starts, gt_ends = target_segs[:, :, :, 0], target_segs[:, :, :, 1]
        negative_offsets = torch.abs(torch.clamp(gt_starts.min(dim=-1)[0], max=0.0))

        query_starts += negative_offsets[:, :, None]
        query_ends += negative_offsets[:, :, None]
        gt_starts += negative_offsets[:, :, None]
        gt_ends += negative_offsets[:, :, None]

        intersect_starts = torch.stack([query_starts, gt_starts], dim=-1).max(dim=-1)[0]        # [B, N_f, N_a]
        intersect_ends = torch.stack([query_ends, gt_ends], dim=-1).min(dim=-1)[0]              # [B, N_f, N_a]
        intersects = torch.clamp(intersect_ends - intersect_starts, min=0.0)

        unions = (gt_ends - gt_starts) + (query_ends - query_starts) - intersects
        return intersects / unions  
    
    def label_queries(self, queries, target, modality, iou_threshold):
        """Label queries that have maximum IOU with ground truth
            segments within the window.

        Args:
            queries (torch.Tensor): Contains the time interval queries for the
            given modality
            target (dict): Contains labels of all the ground truth segments
            within the input window
            modality (str): The modality to get queries from
            iou_threshold (float, optional): IOU threshold with ground truth
            segments to determine valid proposals.

        Returns:
            query_targets (torch.Tensor[B*Nq, 2]): The intervals of the ground
            truth segment the query is trying to regress to
            query_labels (torch.Tensor[B*Nq, Nl]): The labels of the ground
            truth segemnts the query overlaps with. Nl=Number of labels for the
            given modality (3 if including verb/noun in visual, 1 otherwise)
            query_ious (torch.Tensor[B*Nq]): The IOUs with the ground truth
            segment the query is assigned to. Used to reweight CLS loss
        """
        if modality == "visual":
            target_segs = target['v_gt_segments']
            gt_labels = torch.stack([target['verb'], target['noun'], target['action']], dim=-1)
        else:
            target_segs = target['a_gt_segments']
            gt_labels = target['class_id'].unsqueeze(-1)

        # Format queries, targets and labels to calcuate ious
        queries = queries[:, :, None].repeat_interleave(target_segs.shape[1], dim=2)
        target_segs = target_segs[:, None].repeat_interleave(queries.shape[1], dim=1)
        query_labels = gt_labels[:, None].repeat_interleave(queries.shape[1], dim=1)

        ious = self.get_query_ious(queries, target_segs)

        max_iou_inds = ious.argmax(-1).flatten()
        batch_inds = torch.arange(ious.shape[0]).repeat_interleave(ious.shape[1])
        feat_inds = torch.arange(ious.shape[1]).repeat(ious.shape[0])

        ious = ious[batch_inds, feat_inds, max_iou_inds].reshape(ious.shape[0], -1)
        query_targets = target_segs[batch_inds, feat_inds, max_iou_inds].reshape(ious.shape[0], -1, 2)
        query_labels = query_labels[batch_inds, feat_inds, max_iou_inds].reshape(ious.shape[0], -1, query_labels.shape[-1])

        # Fill negative proposals with dummy information
        negatives = (ious < iou_threshold)
        query_targets.masked_fill_(negatives[:, :, None], float("inf"))
        query_labels.masked_fill_(negatives[:, :, None], -1)

        # Flatten all queries in each pyramid level
        query_targets = torch.flatten(query_targets, start_dim=0, end_dim=1)
        query_labels = torch.flatten(query_labels, start_dim=0, end_dim=1)
        query_ious = torch.flatten(ious)

        query_labels = self.assign_positive_labels(modality, query_labels)        

        return query_targets, query_labels, query_ious

    def forward_train(self, inputs, feature_times, target):
        v_offsets = a_offsets = torch.empty(0, 2)
        v_labels = a_labels = torch.empty(0, 4)
        num_v_queries = num_a_queries = 0
        v_queries = a_queries = v_query_ious = a_query_ious = None

        all_times = feature_times

        if "visual" in self.data_modality:
            v_queries = torch.randperm(self.train_pool.shape[1])[:self.num_queries]
            v_queries = self.train_pool[:, v_queries.long()]

            v_queries = v_queries.repeat(all_times.shape[0], 1, 1)
            num_v_queries = v_queries.shape[1]

            v_queries = v_queries.to(device=feature_times.device)
            v_offsets, v_labels, v_query_ious = self.label_queries(
                                            v_queries,
                                            target,
                                            "visual",
                                            self.iou_threshold
                                        )
            all_times = torch.concat([all_times, v_queries], dim=1)
            v_queries = torch.flatten(v_queries, start_dim=0, end_dim=1)

        if "audio" in self.data_modality:
            a_queries = torch.randperm(self.train_pool.shape[1])[:self.num_queries]
            a_queries = self.train_pool[:, a_queries.long()]

            a_queries = a_queries.repeat(all_times.shape[0], 1, 1)
            num_a_queries = a_queries.shape[1]

            a_queries = a_queries.to(device=feature_times.device)
            a_offsets, a_labels, a_query_ious = self.label_queries(
                                            a_queries,
                                            target,
                                            "audio",
                                            self.iou_threshold
                                        )
            all_times = torch.concat([all_times, a_queries], dim=1)
            a_queries = torch.flatten(a_queries, start_dim=0, end_dim=1)

        time_encodings = self.time_mlp(all_times)

        # Project features to lower dim and include time and modality encodings.
        x = self.feature_encoding(inputs, time_encodings, num_v_queries, num_a_queries)

        # Mask queries from each other to remove dependence on queries for performance
        mask = torch.ones(size=(x.size(0), x.size(0)), device=x.device)
        mask[:, :self.num_feats] = 0.
        mask = mask.fill_diagonal_(0.)

        mask = mask.unsqueeze(0)
        mask = mask.repeat_interleave(self.nhead*x.size(1), dim=0).bool()

        x, _ = self.backbone(x, src_mask=mask)

        # Output Shapes: # [B, N_queries, C]
        cls_scores = self.cls_head(x, num_v_queries, num_a_queries)
        reg_scores = self.reg_head(x, num_v_queries, num_a_queries)

        return (cls_scores, reg_scores, x[:, :self.num_feats]), \
                (v_offsets, a_offsets), \
                (v_labels, a_labels), \
                (v_queries, a_queries), \
                (v_query_ious, a_query_ious)

    def forward_inference(self, inputs, feature_times, target, label_queries=False):
        v_offsets = a_offsets = torch.empty(0, 2)
        v_labels = a_labels = torch.empty(0, 4)
        num_v_queries = num_a_queries = 0
        v_queries = a_queries = v_query_ious = a_query_ious = None

        all_times = feature_times

        if "visual" in self.data_modality:
            v_queries = self.inference_queries.repeat(all_times.shape[0], 1, 1)
            num_v_queries = v_queries.shape[1]

            v_queries = v_queries.to(device=feature_times.device)
            if label_queries:
                v_offsets, v_labels, v_query_ious = self.label_queries(
                                                v_queries,
                                                target,
                                                "visual",
                                                self.iou_threshold
                                            )
            all_times = torch.concat([all_times, v_queries], dim=1)
            v_queries = torch.flatten(v_queries, start_dim=0, end_dim=1)

        if "audio" in self.data_modality:
            a_queries = torch.randperm(self.inference_queries.shape[1])[:self.num_queries]
            a_queries = self.inference_queries.repeat(all_times.shape[0], 1, 1)
            num_a_queries = a_queries.shape[1]

            a_queries = a_queries.to(device=feature_times.device)
            if label_queries:
                a_offsets, a_labels, a_query_ious = self.label_queries(
                                                a_queries,
                                                target,
                                                "audio",
                                                self.iou_threshold
                                            )
            all_times = torch.concat([all_times, a_queries], dim=1)
            a_queries = torch.flatten(a_queries, start_dim=0, end_dim=1)

        time_encodings = self.time_mlp(all_times)

        # Project features to lower dim and include time and modality encodings.
        x = self.feature_encoding(inputs, time_encodings, num_v_queries, num_a_queries)

        # Mask queries from each other to remove dependence on queries for performance
        masks = torch.ones(size=(x.size(0), x.size(0)), device=x.device)
        masks[:, :self.num_feats] = 0.
        masks = masks.fill_diagonal_(0.)

        masks = masks.unsqueeze(0)
        masks = masks.repeat_interleave(self.nhead*x.size(1), dim=0).bool()
        x, _ = self.backbone(x, src_mask=masks)

        # Output Shapes: # [B, N_queries, C]
        cls_scores = self.cls_head(x, num_v_queries, num_a_queries)
        reg_scores = self.reg_head(x, num_v_queries, num_a_queries)

        return (cls_scores, reg_scores, x[:, :self.num_feats]), \
                (v_offsets, a_offsets), \
                (v_labels, a_labels), \
                (v_queries, a_queries), \
                (v_query_ious, a_query_ious)

    def forward_encoder(
                self,
                inputs,
                feature_times,
                target,
                label_queries=False
            ):
        if self.training:
            return self.forward_train(inputs, feature_times, target)
        else:
            return self.forward_inference(inputs, feature_times, target, label_queries)


    def forward(self,
                inputs,
                forward_type,
                feature_times=None,
                target=None,
                label_queries=False
            ):
        if forward_type == "encoder":
            return self.forward_encoder(
                                    inputs,
                                    feature_times,
                                    target,
                                    label_queries
                                )
        elif forward_type == "drloc_mlp":
            return self.drloc_mlp(inputs).squeeze(2)
