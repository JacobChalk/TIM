#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
import sys

from .omnivore_model import omnivore_swinB_epic, omnivore_swinB
from .build import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class Omnivore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #self.omni = torch.hub.load("facebookresearch/omnivore", model=cfg.MODEL.ARCH, force_reload=True)
        if cfg.MODEL.ARCH == 'omnivore_swinB_epic':
            self.omni = omnivore_swinB_epic()
        elif cfg.MODEL.ARCH == 'omnivore_swinB':
            self.omni = omnivore_swinB(pretrained=True, load_heads=True)
        else:
            print("no such architecture : ", cfg.MODEL.ARCH)
            sys.exit(1)

        # replace the last head to identity
        self.omni.heads = nn.Identity()
        #self.register_buffer('verb_matrix',self._get_output_transform_matrix('verb',cfg))
        #self.register_buffer('noun_matrix',self._get_output_transform_matrix('noun',cfg))

    # def _get_output_transform_matrix(self, which_one,cfg):

    #     with open('slowfast/models/omnivore_epic_action_classes.csv') as f:
    #         data = f.read().splitlines()
    #         action2index = {d:i for i,d in enumerate(data)}


    #     if which_one == 'verb':
    #         verb_classes = pd.read_csv(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/EPIC_100_verb_classes.csv',usecols=['key'])
    #         verb2index = {}
    #         for verb in verb_classes['key']:
    #             verb2index[verb] = [v for k,v in action2index.items() if k.split(',')[0]==verb]
    #         matrix = torch.zeros(len(action2index),len(verb2index))
    #         for i, (k,v) in enumerate(verb2index.items()):
    #             for j in v:
    #                 matrix[j,i] = 1.
    #     elif which_one == 'noun':
    #         noun_classes = pd.read_csv(f'{cfg.EPICKITCHENS.ANNOTATIONS_DIR}/EPIC_100_noun_classes.csv',usecols=['key'])
    #         noun2index = {}
    #         for noun in noun_classes['key']:
    #             noun2index[noun] = [v for k,v in action2index.items() if k.split(',')[1]==noun]
    #         matrix = torch.zeros(len(action2index),len(noun2index))
    #         for i, (k,v) in enumerate(noun2index.items()):
    #             for j in v:
    #                 matrix[j,i] = 1.
    #     return matrix


    def forward(self, x):
        y = self.omni(x, input_type="video")
        return y