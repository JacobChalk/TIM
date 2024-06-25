#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import os
import torch
from fvcore.common.file_io import PathManager

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import FeatureMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg):
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    for cur_iter, (inputs, labels, audio_idx, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()
            audio_idx = audio_idx.cuda()
        test_meter.data_toc()

        # Perform the forward pass.
        preds, feature = model(inputs)

        if isinstance(labels, (dict,)):
            # Gather all the predictions across all the devices to perform ensemble.
            # You don't need to use preds or labels but just in case if you want to evaluate the model, we'll keep it.
            if cfg.NUM_GPUS > 1:
                verb_preds, verb_labels, audio_idx = du.all_gather(
                    [preds[0], labels['verb'], audio_idx]
                )

                noun_preds, noun_labels, audio_idx = du.all_gather(
                    [preds[1], labels['noun'], audio_idx]
                )
                feature, audio_idx = du.all_gather(
                    [feature, audio_idx]
                )
                meta = du.all_gather_unaligned(meta)
                metadata = {'narration_id': []}
                for i in range(len(meta)):
                    metadata['narration_id'].extend(meta[i]['narration_id'])
            else:
                metadata = meta
                verb_preds, verb_labels, audio_idx = preds[0], labels['verb'], audio_idx
                noun_preds, noun_labels, audio_idx = preds[1], labels['noun'], audio_idx

            if cfg.NUM_GPUS:
                verb_preds = verb_preds.cpu()
                verb_labels = verb_labels.cpu()
                noun_preds = noun_preds.cpu()
                noun_labels = noun_labels.cpu()
                audio_idx = audio_idx.cpu()
                feature = feature.cpu()

            test_meter.iter_toc()

            # Update and log features.
            test_meter.update_features(
                metadata, feature.detach(),
                audio_idx.detach(),
            )

            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    metadata, features = test_meter.finalize_features()
    return metadata, features


def test(cfg):
    """
    Perform multi-view testing on the pretrained audio model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the audio model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg)

    cu.load_test_checkpoint(cfg, model)

    # Create audio testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % cfg.TEST.NUM_FEATURES
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = FeatureMeter(
        len(test_loader.dataset)
        // cfg.TEST.NUM_FEATURES,
        cfg.TEST.NUM_FEATURES,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader)
    )


    # # Perform multi-view test on the entire dataset.
    metadata, audio_features = perform_test(test_loader, model, test_meter, cfg)

    # Save the meta data
    featurefile = os.path.join(cfg.OUTPUT_DIR, 'features.npy')
    np.save(featurefile, audio_features)
    metafile = os.path.join(cfg.OUTPUT_DIR, 'metadata.npy')
    np.save(metafile, metadata)
