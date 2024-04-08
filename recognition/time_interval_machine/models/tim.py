import torch
import random

from torch import nn

import time_interval_machine.models.helpers.head as head
import time_interval_machine.utils.logging as logging


from time_interval_machine.models.helpers.encodings import AudioVisualFeatureEncoding, VisualFeatureEncoding, AudioFeatureEncoding
from time_interval_machine.models.helpers.transformers import TransformerEncoder, TransformerEncoderLayer
from time_interval_machine.models.helpers.pool import AVGA


logger = logging.get_logger(__name__)

class TIM(nn.Module):
    def __init__(self,
                num_class,
                visual_input_dim=1024,
                audio_input_dim=2304,
                feat_drop=0.5,
                seq_drop=0.5,
                d_model=512,
                feedforward_scale=4,
                nhead=8,
                num_layers=6,
                enc_dropout=0.1,
                input_modality="audio_visual",
                data_modality="audio_visual",
                num_feats=50,
                include_verb_noun=True,
                pool_features=False
            ):
        super(TIM, self).__init__()

        self.input_modality = input_modality
        self.data_modality = data_modality

        self.visual_input_dim = visual_input_dim
        self.audio_input_dim = audio_input_dim
        self.feat_drop=feat_drop
        self.seq_drop = seq_drop

        self.d_model = d_model
        self.dim_feedforward = d_model*feedforward_scale
        self.nhead = nhead
        self.num_layers = num_layers
        self.enc_dropout = enc_dropout

        self.num_feats = num_feats
        self.num_class = num_class
        self.include_verb_noun = include_verb_noun
        self.pool_features = pool_features

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
                                        self.data_modality,
                                        self.include_verb_noun
                                    )
            self.num_feats *= 2
        elif self.input_modality == "visual":
            self.feature_encoding = VisualFeatureEncoding(
                                        self.visual_input_dim,
                                        self.d_model,
                                        self.feat_drop,
                                        self.seq_drop,
                                        self.num_feats,
                                        self.include_verb_noun
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
        elif self.data_modality == "visual":
            self.cls_head = head.VisualCLSHead(self.num_class[0], 2*self.d_model)
        else:
            self.cls_head = head.AudioCLSHead(self.num_class[1], 2*self.d_model)

        encoder_layer = TransformerEncoderLayer(
                            d_model=2*self.d_model,
                            nhead=self.nhead,
                            dim_feedforward=self.dim_feedforward,
                            dropout=self.enc_dropout,
                            activation='gelu'
                        )

        self.transformer_encoder = TransformerEncoder(
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

        if self.pool_features:
            self.pool = AVGA(
                    a_dim=self.audio_input_dim,
                    v_dim=self.visual_input_dim,
                    hidden_size=self.visual_input_dim
            )
        else:
            self.pool = None


    def forward_encoder(
                self,
                inputs,
                time_encodings,
                num_v_queries,
                num_a_queries
            ):
        
        if self.pool_features:
            inputs[0] = self.pool(inputs[1], inputs[0])

        # Project features to lower dim and include time and modality encodings
        x = self.feature_encoding(inputs, time_encodings, num_v_queries, num_a_queries) # Shape: [S, B, C]

        masks = torch.ones(size=(x.size(0), x.size(0)), device=x.device)
        masks[:, :self.num_feats] = 0.
        masks = masks.fill_diagonal_(0.)

        masks = masks.unsqueeze(0)
        masks = masks.repeat_interleave(self.nhead*x.size(1), dim=0).bool()  # Masks Shape: [B*n, S, S]

        x, _ = self.transformer_encoder(x, src_mask=masks)          # Shape: [B, S, C]

        cls_scores = self.cls_head(x, num_v_queries, num_a_queries) # Shape: [B, Nq, C]

        return (cls_scores, x[:, :self.num_feats])

    def forward(self,
                inputs,
                forward_type,
                time_encodings=None,
                num_v_queries=None,
                num_a_queries=None
            ):
        if forward_type == "time_mlp":
            return self.time_mlp(inputs)
        elif forward_type == "encoder":
            return self.forward_encoder(
                                    inputs,
                                    time_encodings,
                                    num_v_queries,
                                    num_a_queries
                                )
        elif forward_type == "drloc_mlp":
            return self.drloc_mlp(inputs).squeeze(2)

