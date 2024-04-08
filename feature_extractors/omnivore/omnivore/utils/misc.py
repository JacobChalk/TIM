#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import numpy as np
import os
from datetime import datetime
import torch
from fvcore.nn.flop_count import flop_count
from torch import nn

import omnivore.utils.logging as logging
from omnivore.datasets.utils import pack_pathway_output

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def get_flop_stats(model, cfg, is_train):
    """
    Compute the gflops for the current model given the config.
    Args:
        model (model): model to compute the flop counts.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, compute flops for training. Otherwise,
            compute flops for testing.

    Returns:
        float: the total number of gflops of the given model.
    """
    rgb_dimension = 3
    #if is_train:
    #    input_tensors = torch.rand(
    #        rgb_dimension,
    #        cfg.DATA.NUM_FRAMES,
    #        cfg.DATA.TRAIN_CROP_SIZE,
    #        cfg.DATA.TRAIN_CROP_SIZE,
    #    )
    #else:
    input_tensors = torch.rand(
        rgb_dimension,
        cfg.DATA.NUM_FRAMES,
        cfg.DATA.TEST_CROP_SIZE,
        cfg.DATA.TEST_CROP_SIZE,
    )

    flop_inputs = pack_pathway_output(cfg, input_tensors)
    for i in range(len(flop_inputs)):
        flop_inputs[i] = flop_inputs[i].unsqueeze(0).cuda(non_blocking=True)

    # If detection is enabled, count flops for one proposal.
    #if cfg.DETECTION.ENABLE:
    #    bbox = torch.tensor([[0, 0, 1.0, 0, 1.0]])
    #    bbox = bbox.cuda()
    #    inputs = (flop_inputs, bbox)
    #else:
    inputs = (flop_inputs,)

    gflop_dict, _ = flop_count(model, inputs)
    gflops = sum(gflop_dict.values())
    return gflops


def log_model_info(model, cfg, is_train=True):
    """
    Log info, includes number of parameters, gpu usage and gflops.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        is_train (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info(
        "FLOPs: {:,} GFLOPs".format(get_flop_stats(model, cfg, is_train))
    )
    logger.info("nvidia-smi")
    os.system("nvidia-smi")


def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (int): current epoch.
    """
    return (
        cur_epoch + 1
    ) % cfg.TRAIN.EVAL_PERIOD == 0 or cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()
