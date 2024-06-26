# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified from https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/generic/distributed_util.py

import io
import os
import tempfile
from typing import Any, Callable, List, Tuple, Dict, Optional

import torch
import torch.distributed as dist


from iopath.common.file_io import g_pathmgr


_PRIMARY_RANK = 0


def is_local_primary():
    return int(os.getenv("LOCAL_RANK")) == 0


def is_local_primary_cuda():
    assert dist.is_initialized()
    assert torch.cuda.is_available()
    return torch.cuda.current_device() == 0


def is_torch_dataloader_worker():
    return torch.utils.data.get_worker_info() is not None


def convert_to_distributed_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor: torch.Tensor, orig_device: str) -> torch.Tensor:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor


def is_distributed_training_run() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


def is_primary() -> bool:
    """
    Returns True if this is rank 0 of a distributed training job OR if it is
    a single trainer job. Otherwise False.
    """
    return get_rank() == _PRIMARY_RANK


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing mean reduction
    of tensor over all processes.
    """
    return all_reduce_op(
        tensor,
        torch.distributed.ReduceOp.SUM,
        lambda t: t / torch.distributed.get_world_size(),
    )


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing sum
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.SUM)


def all_reduce_min(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing min
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MIN)


def all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing min
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MAX)


def all_reduce_op(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    after_op_func: Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.all_reduce(tensor, op)
        if after_op_func is not None:
            tensor = after_op_func(tensor)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def gather_tensors_from_all(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    gathered_tensors = gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Wrapper over torch.distributed.broadcast for broadcasting a tensor from the source
    to all processes in both distributed / non-distributed scenarios.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.broadcast(tensor, src)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def barrier() -> None:
    """
    Wrapper over torch.distributed.barrier, returns without waiting
    if the distributed process group is not initialized instead of throwing error.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    torch.distributed.barrier()


def get_world_size() -> int:
    """
    Simple wrapper for correctly getting worldsize in both distributed
    / non-distributed settings
    """
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )


def get_rank() -> int:
    """
    Simple wrapper for correctly getting rank in both distributed
    / non-distributed settings
    """
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )


def broadcast_object(obj: Any, src: int = _PRIMARY_RANK, use_disk: bool = True) -> Any:
    """Broadcast an object from a source to all workers.

    Args:
        obj: Object to broadcast, must be serializable
        src: Source rank for broadcast (default is primary)
        use_disk: If enabled, removes redundant CPU memory copies by writing to
            disk
    """
    # Either broadcast from primary to the fleet (default),
    # or use the src setting as the original rank
    if get_rank() == src:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data_view = buffer.getbuffer()
        length_tensor = torch.LongTensor([len(data_view)])
        length_tensor = broadcast(length_tensor, src=src)
        data_tensor = torch.ByteTensor(data_view)
        data_tensor = broadcast(data_tensor, src=src)
    else:
        # Fetch from the source
        length_tensor = torch.LongTensor([0])
        length_tensor = broadcast(length_tensor, src=src)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8)
        data_tensor = broadcast(data_tensor, src=src)
        if use_disk:
            with tempfile.TemporaryFile("r+b") as f:
                f.write(data_tensor.numpy())
                # remove reference to the data tensor and hope that Python garbage
                # collects it
                del data_tensor
                f.seek(0)
                obj = torch.load(f)
        else:
            buffer = io.BytesIO(data_tensor.numpy())
            obj = torch.load(buffer)
    return obj

    import logging

# constants:
CHECKPOINT_FILE = "checkpoint.torch"
CPU_DEVICE = torch.device("cpu")
GPU_DEVICE = torch.device("cuda")


def load_and_broadcast_checkpoint_list(
    checkpoint_paths: List[str], device: torch.device = CPU_DEVICE
):
    if is_primary():
        for path in checkpoint_paths:
            checkpoint = load_checkpoint(path, device)
            if checkpoint is not None:
                break
    else:
        checkpoint = None
    logging.info(f"Broadcasting checkpoint loaded from {checkpoint_paths}")
    return broadcast_object(checkpoint)


def load_and_broadcast_checkpoint(
    checkpoint_path: str, device: torch.device = CPU_DEVICE
) -> Optional[Dict]:
    """Loads a checkpoint on primary and broadcasts it to all replicas.

    This is a collective operation which needs to be run in sync on all replicas.

    See :func:`load_checkpoint` for the arguments.
    """
    if is_primary():
        checkpoint = load_checkpoint(checkpoint_path, device)
    else:
        checkpoint = None
    logging.info(f"Broadcasting checkpoint loaded from {checkpoint_path}")
    return broadcast_object(checkpoint)


def load_checkpoint(
    checkpoint_path: str, device: torch.device = CPU_DEVICE
) -> Optional[Dict]:
    """Loads a checkpoint from the specified checkpoint path.

    Args:
        checkpoint_path: The path to load the checkpoint from. Can be a file or a
            directory. If it is a directory, the checkpoint is loaded from
            :py:data:`CHECKPOINT_FILE` inside the directory.
        device: device to load the checkpoint to

    Returns:
        The checkpoint, if it exists, or None.
    """
    if not checkpoint_path:
        return None

    assert device is not None, "Please specify what device to load checkpoint on"
    assert device.type in ["cpu", "cuda"], f"Unknown device: {device}"
    if device.type == "cuda":
        assert torch.cuda.is_available()

    if not g_pathmgr.exists(checkpoint_path):
        logging.warning(f"Checkpoint path {checkpoint_path} not found")
        return None
    if g_pathmgr.isdir(checkpoint_path):
        checkpoint_path = f"{checkpoint_path.rstrip('/')}/{CHECKPOINT_FILE}"

    if not g_pathmgr.exists(checkpoint_path):
        logging.warning(f"Checkpoint file {checkpoint_path} not found.")
        return None

    logging.info(f"Attempting to load checkpoint from {checkpoint_path}")
    # load model on specified device and not on saved device for model and return
    # the checkpoint
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=device)
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint