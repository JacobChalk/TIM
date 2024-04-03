import torch.nn as nn
import torch

class AudioVisualCLSHead(nn.Module):
    def __init__(self, num_class, d_model):
        super(AudioVisualCLSHead, self).__init__()
        if isinstance(num_class[0], list):
            self.fc_visual_verb = nn.Linear(d_model, num_class[0][0])
            self.fc_visual_noun = nn.Linear(d_model, num_class[0][1])
            self.fc_visual_action = nn.Linear(d_model, num_class[0][2])
            self.include_verb_noun = True
        else:
            self.fc_visual_action = nn.Linear(d_model, num_class[0])
            self.include_verb_noun = False
        self.fc_audio_action = nn.Linear(d_model, num_class[1])

    def forward(self, x, num_v_queries, num_a_queries):
        aud_start = -num_a_queries if num_a_queries > 0 else x.size(1)
        action_start = aud_start - num_v_queries

        if self.include_verb_noun:
            noun_start = action_start - num_v_queries
            verb_start = noun_start - num_v_queries
            v_verb_preds = self.fc_visual_verb(x[:, verb_start:noun_start])
            v_noun_preds = self.fc_visual_noun(x[:, noun_start:action_start])
            v_verb_preds = torch.flatten(v_verb_preds, start_dim=0, end_dim=1)
            v_noun_preds = torch.flatten(v_noun_preds, start_dim=0, end_dim=1)
        else:
            v_verb_preds = None
            v_noun_preds = None

        v_action_preds = self.fc_visual_action(x[:, action_start:aud_start])
        v_action_preds = torch.flatten(v_action_preds, start_dim=0, end_dim=1)

        a_action_preds = self.fc_audio_action(x[:, aud_start:])
        a_action_preds = torch.flatten(a_action_preds, start_dim=0, end_dim=1)

        return (v_verb_preds, v_noun_preds, v_action_preds, a_action_preds)

class VisualCLSHead(nn.Module):
    def __init__(self, num_class, d_model):
        super(VisualCLSHead, self).__init__()
        if isinstance(num_class, list):
            self.fc_visual_verb = nn.Linear(d_model, num_class[0])
            self.fc_visual_noun = nn.Linear(d_model, num_class[1])
            self.fc_visual_action = nn.Linear(d_model, num_class[2])
            self.include_verb_noun = True
        else:
            self.fc_visual_action = nn.Linear(d_model, num_class)
            self.include_verb_noun = False


    def forward(self, x, num_v_queries, num_a_queries):
        action_start = -num_v_queries
        if self.include_verb_noun:
            noun_start = action_start - num_v_queries
            verb_start = noun_start - num_v_queries
            v_verb_preds = self.fc_visual_verb(x[:, verb_start:noun_start])
            v_noun_preds = self.fc_visual_noun(x[:, noun_start:action_start])
            v_verb_preds = torch.flatten(v_verb_preds, start_dim=0, end_dim=1)
            v_noun_preds = torch.flatten(v_noun_preds, start_dim=0, end_dim=1)
        else:
            v_verb_preds = None
            v_noun_preds = None

        v_action_preds = self.fc_visual_action(x[:, action_start:])
        v_action_preds = torch.flatten(v_action_preds, start_dim=0, end_dim=1)

        return (v_verb_preds, v_noun_preds, v_action_preds, None)

class AudioCLSHead(nn.Module):
    def __init__(self, num_class, d_model):
        super(AudioCLSHead, self).__init__()
        self.fc_audio_action = nn.Linear(d_model, num_class)

    def forward(self, x, num_v_queries, num_a_queries):
        a_action_preds = self.fc_audio_action(x[:, -num_a_queries:])

        a_action_preds = torch.flatten(a_action_preds, start_dim=0, end_dim=1)

        return (None, None, None, a_action_preds)