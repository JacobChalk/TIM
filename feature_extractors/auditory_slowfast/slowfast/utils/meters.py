#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import torch
from fvcore.common.timer import Timer

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class FeatureMeter(object):

    def __init__(
            self,
            num_audios,
            num_clips,
            num_cls,
            overall_iters,
    ):

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters

        # Initialize tensors.
        self.metadata = np.zeros(num_audios, dtype=object)
        self.clip_count = torch.zeros((num_audios)).long()
        self.topk_accs = []
        self.stats = {}

        # For feature extraction
        self.audio_features = torch.zeros((num_audios, num_clips, 2304))  # 2304 is the output dimension of EPIC_SOUNDS

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.metadata.fill(0)

        # For feature extraction
        self.audio_features.zero_()

    def update_features(self, metadata, features, clip_ids):
        for ind in range(features.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            clip_temporal_id = int(clip_ids[ind]) % self.num_clips

            self.audio_features[vid_id, clip_temporal_id] = features[ind]
            self.metadata[vid_id] = metadata['narration_id'][ind]
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_features(self):
        return self.metadata.copy(), self.audio_features.numpy().copy()