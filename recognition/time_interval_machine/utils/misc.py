import numpy as np
import subprocess
import argparse
import psutil
import torch
import math
import os

from datetime import datetime
from pathlib import Path

import time_interval_machine.utils.multiprocessing as mpu

import time_interval_machine.utils.logging as logging

logger = logging.get_logger(__name__)

def check_nan(t):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if torch.isnan(t).any():
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))

def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    total = torch.cuda.mem_get_info()[1]
    return mem_usage_bytes / 1024 ** 3, total / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def cache_data_triton(path_to_dataset, path_to_file, machine):
    """
    When running jobs on triton/slurm, automatically cache
    data in the temp folder.
    """
    if machine not in ["triton", "slurm"]:
        return os.path.join(path_to_dataset, path_to_file)
    caching_location = os.path.join("/tmp/", path_to_file)
    if not os.path.exists(caching_location):
        Path(caching_location).parent.mkdir(parents=True, exist_ok=True)
        subprocess.call(f'cp {os.path.join(path_to_dataset, path_to_file)} {caching_location}', shell=True)

    return caching_location

def is_overlap(segment1, segment2):
    return ((segment1[0] <= segment2[1]) and (segment2[0] <= segment1[1]))

def launch_job(args, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if args.num_gpus > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=args.num_gpus,
            args=(
                args.num_gpus,
                func,
                init_method,
                args.shard_id,
                args.num_shards,
                args.dist_backend,
                args
            ),
            daemon=daemon,
        )
    else:
        func(args=args)


def process_metadata(metadata):
    a_queries = torch.max(metadata['num_a_queries']).item()
    v_queries = torch.max(metadata['num_v_queries']).item()

    for k, v in metadata.items():
        if 'a_' in k:
            if isinstance(v, list):
                metadata[k] = np.array(v).T.flatten()
            elif isinstance(v, torch.Tensor):
                metadata[k] = torch.flatten(v).cuda(non_blocking=True)

    for k, v in metadata.items():
        if 'v_' in k:
            if isinstance(v, list):
                metadata[k] = np.array(v).T.flatten()
            elif isinstance(v, torch.Tensor):
                metadata[k] = torch.flatten(v).cuda(non_blocking=True)

    return metadata, v_queries, a_queries