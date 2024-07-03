#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import os
import numpy as np
import torch

import omnivore.utils.distributed as du
import omnivore.utils.logging as logging
from omnivore.datasets import loader
from omnivore.models import build_model
from omnivore.utils.meters import TestFeatureMeter

logger = logging.get_logger(__name__)


def perform_feature_extraction(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):

        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        video_idx = video_idx.cuda()

        # Perform the forward pass.
        preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, video_idx = du.all_gather(
                [preds, video_idx]
            )
            meta = du.all_gather_unaligned(meta)
            metadata = {'narration_id': []}
            for i in range(len(meta)):
                metadata['narration_id'].extend(meta[i]['narration_id'])
        else:
            metadata = meta
        test_meter.iter_toc()

        # Update and log stats.
        test_meter.update_stats(
            preds.detach().cpu(),
            metadata,
            video_idx.detach(),
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    features, metadata = test_meter.finalize_features()

    test_meter.reset()
    return features, metadata

def extract_features(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if not cfg.TEST.SLIDE.ENABLE:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_FEATURES * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )

    # Create meters
    test_meter = TestFeatureMeter(
        len(test_loader.dataset),
        cfg.TEST.NUM_FEATURES, # each window plays an equal weight on the metrics
        len(test_loader)
    )

    # Perform multi-view test on the entire dataset.
    features, metadata = perform_feature_extraction(test_loader, model, test_meter, cfg)

    if du.is_master_proc():
        output_dir = os.path.join(cfg.OUTPUT_DIR, 'features')
        os.makedirs(output_dir, exist_ok=True)
        feature_file = os.path.join(output_dir, 'features.npy')
        np.save(feature_file, features)
