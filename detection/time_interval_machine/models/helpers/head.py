import torch.nn.functional as F
import torch.nn as nn
import torch

import math

class AudioVisualCLSHead(nn.Module):
    def __init__(self, num_class, d_model):
        super(AudioVisualCLSHead, self).__init__()
        self.include_verb_noun = isinstance(num_class, list)

        bias_value = -(math.log((1 - 0.01) / 0.01))
        if self.include_verb_noun:
            self.fc_visual_verb = nn.Linear(d_model, num_class[0][0])
            self.fc_visual_noun = nn.Linear(d_model, num_class[0][1])
            self.fc_visual_action = nn.Linear(d_model, num_class[0][2])

            torch.nn.init.constant_(self.fc_visual_verb.bias, bias_value)
            torch.nn.init.constant_(self.fc_visual_noun.bias, bias_value)
        else:
            self.fc_visual_action = nn.Linear(d_model, num_class[0])

        self.fc_audio_action = nn.Linear(d_model, num_class[1])
        torch.nn.init.constant_(self.fc_visual_action.bias, bias_value)
        torch.nn.init.constant_(self.fc_audio_action.bias, bias_value)

    def forward(self, x, num_v_queries, num_a_queries):
        audio_start = -num_a_queries if num_a_queries > 0 else x.size(1)
        visual_start = audio_start - num_v_queries
        v_verb_preds = None
        v_noun_preds = None

        if self.include_verb_noun:
            v_verb_preds = self.fc_visual_verb(x[:, visual_start:audio_start])
            v_noun_preds = self.fc_visual_noun(x[:, visual_start:audio_start])

            v_verb_preds = torch.flatten(v_verb_preds, start_dim=0, end_dim=1)
            v_noun_preds = torch.flatten(v_noun_preds, start_dim=0, end_dim=1)

        v_action_preds = self.fc_visual_action(x[:, visual_start:audio_start])
        a_action_preds = self.fc_audio_action(x[:, audio_start:])

        v_action_preds = torch.flatten(v_action_preds, start_dim=0, end_dim=1)
        a_action_preds = torch.flatten(a_action_preds, start_dim=0, end_dim=1)

        return (v_verb_preds, v_noun_preds, v_action_preds, a_action_preds)

class VisualCLSHead(nn.Module):
    def __init__(self, num_class, d_model):
        super(VisualCLSHead, self).__init__()
        self.include_verb_noun = isinstance(num_class, list)

        bias_value = -(math.log((1 - 0.01) / 0.01))
        if self.include_verb_noun:
            self.fc_visual_verb = nn.Linear(d_model, num_class[0])
            self.fc_visual_noun = nn.Linear(d_model, num_class[1])
            self.fc_visual_action = nn.Linear(d_model, num_class[2])
            torch.nn.init.constant_(self.fc_visual_verb.bias, bias_value)
            torch.nn.init.constant_(self.fc_visual_noun.bias, bias_value)
        else:
            self.fc_visual_action = nn.Linear(d_model, num_class)

        torch.nn.init.constant_(self.fc_visual_action.bias, bias_value)

    def forward(self, x, num_v_queries, num_a_queries):
        v_verb_preds = None
        v_noun_preds = None

        if self.include_verb_noun:
            v_verb_preds = self.fc_visual_verb(x[:, -num_v_queries:])
            v_noun_preds = self.fc_visual_noun(x[:, -num_v_queries:])

            v_verb_preds = torch.flatten(v_verb_preds, start_dim=0, end_dim=1)
            v_noun_preds = torch.flatten(v_noun_preds, start_dim=0, end_dim=1)

        v_action_preds = self.fc_visual_action(x[:, -num_v_queries:])
        v_action_preds = torch.flatten(v_action_preds, start_dim=0, end_dim=1)

        return (v_verb_preds, v_noun_preds, v_action_preds, None)

class AudioCLSHead(nn.Module):
    def __init__(self, num_class, d_model):
        super(AudioCLSHead, self).__init__()

        bias_value = -(math.log((1 - 0.01) / 0.01))
        self.fc_audio_action = nn.Linear(d_model, num_class)
        torch.nn.init.constant_(self.fc_audio_action.bias, bias_value)

    def forward(self, x, num_v_queries, num_a_queries):
        a_action_preds = self.fc_audio_action(x[:, -num_a_queries:])
        a_action_preds = torch.flatten(a_action_preds, start_dim=0, end_dim=1)

        return (None, None, None, a_action_preds)

class AudioVisualRegHead(nn.Module):
    def __init__(self, d_model):
        super(AudioVisualRegHead, self).__init__()

        self.fc_visual_action = nn.Sequential(
                            nn.Linear(d_model, d_model // 2),
                            nn.ReLU(),
                            nn.Linear(d_model // 2, d_model // 2),
                            nn.ReLU(),
                            nn.Linear(d_model // 2, 2),
                            nn.Sigmoid()
                        )
        self.fc_audio_action = nn.Sequential(
                            nn.Linear(d_model, d_model // 2),
                            nn.ReLU(),
                            nn.Linear(d_model // 2, d_model // 2),
                            nn.ReLU(),
                            nn.Linear(d_model // 2, 2),
                            nn.Sigmoid()
                        )

    def forward(self, x, num_v_queries, num_a_queries):
        audio_start = -num_a_queries if num_a_queries > 0 else x.size(1)
        visual_start = audio_start - num_v_queries

        v_action_preds = self.fc_visual_action(x[:, visual_start:audio_start])
        a_action_preds = self.fc_audio_action(x[:, audio_start:])

        v_action_preds = torch.flatten(v_action_preds, start_dim=0, end_dim=1)
        a_action_preds = torch.flatten(a_action_preds, start_dim=0, end_dim=1)

        return (v_action_preds, a_action_preds)

class VisualRegHead(nn.Module):
    def __init__(self, d_model):
        super(VisualRegHead, self).__init__()

        self.fc_visual_action = nn.Sequential(
                            nn.Linear(d_model, d_model // 2),
                            nn.ReLU(),
                            nn.Linear(d_model // 2, d_model // 2),
                            nn.ReLU(),
                            nn.Linear(d_model // 2, 2),
                            nn.Sigmoid()
                        )

    def forward(self, x, num_v_queries, num_a_queries):
        v_action_preds = self.fc_visual_action(x[:, -num_v_queries:])
        v_action_preds = torch.flatten(v_action_preds, start_dim=0, end_dim=1)

        return (v_action_preds, None)

class AudioRegHead(nn.Module):
    def __init__(self, d_model):
        super(AudioRegHead, self).__init__()
        self.fc_audio_action = nn.Sequential(
                            nn.Linear(d_model, d_model // 2),
                            nn.ReLU(),
                            nn.Linear(d_model // 2, d_model // 2),
                            nn.ReLU(),
                            nn.Linear(d_model // 2, 2),
                            nn.Sigmoid()
                        )

    def forward(self, x, num_v_queries, num_a_queries):
        a_action_preds = self.fc_audio_action(x[:, -num_a_queries:])
        a_action_preds = torch.flatten(a_action_preds, start_dim=0, end_dim=1)

        return (None, a_action_preds)
