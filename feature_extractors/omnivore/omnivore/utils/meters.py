#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jaesung Huh, 2024

import datetime
import numpy as np
import torch
from fvcore.common.timer import Timer

import omnivore.utils.logging as logging

logger = logging.get_logger(__name__)


class TestFeatureMeter(object):
    def __init__(self, num_videos, num_clips, overall_iters, feature_dim=1024):
        """
        Construct tensors to store the features. Expect to get
        num_clips predictions from each video, and calculate the features on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            feature_dim (int): feature dimension for each clip.
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.features = torch.zeros(num_videos, self.num_clips, feature_dim)  # Omnivore feature dimension : 1024
        self.metadata = np.zeros(num_videos, dtype=object)
        self.clip_count = torch.zeros((num_videos)).long()

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.features.zero_()
        self.metadata.fill(0)

    def update_stats(self, preds, metadata, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            clip_id = int(clip_ids[ind]) % self.num_clips
            self.features[vid_id, clip_id] = preds[ind]
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
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_features(self):
        """
        Calculate and log the final features.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    self.clip_count, self.num_clips
                )
            )
            logger.warning(self.clip_count)

        return (self.features.cpu().numpy().copy(), self.metadata.copy())

if __name__ == '__main__':
    class2 = TestFeatureMeter(1,1,1,1)