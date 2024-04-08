import torch

from torch.utils.data.distributed import DistributedSampler

import time_interval_machine.utils.logging as logging

from time_interval_machine.datasets.sliding_window import SlidingWindowDataset


logger = logging.get_logger(__name__)

def create_loader(args, split, modality, generator):
    logger.info("Creating {} loader for modality: {}".format(split, modality))
    if split == "train":
        v_action_pkl = args.video_train_action_pickle
        a_action_pkl = args.audio_train_action_pickle
        v_context_pkl = args.video_train_context_pickle
        a_context_pkl = args.audio_train_context_pickle
        shuffle = True
    else:
        v_action_pkl = args.video_val_action_pickle
        a_action_pkl = args.audio_val_action_pickle
        v_context_pkl = args.video_val_context_pickle
        a_context_pkl = args.audio_val_context_pickle
        shuffle = False

    dataset = SlidingWindowDataset(
                    args.video_data_path,
                    args.audio_data_path,
                    v_action_pkl,
                    a_action_pkl,
                    v_context_pkl,
                    a_context_pkl,
                    args.video_info_pickle,
                    v_feature_dim=args.visual_input_dim,
                    a_feature_dim=args.audio_input_dim,
                    num_feats=args.num_feats,
                    feat_stride=args.feat_stride,
                    feat_gap=args.feat_gap,
                    window_stride=args.window_stride,
                    mode=split,
                    data_modality=modality,
                    model_modality=args.model_modality,
                    include_verb_noun=args.include_verb_noun,
                    dataset_name=args.dataset
                )

    batch_size = int(args.batch_size / max(1, args.num_gpus))
    workers = int(args.workers / max(1, args.num_gpus))
    sampler = DistributedSampler(dataset) if args.num_gpus > 1 else None
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=workers,
            pin_memory=args.pin_memory,
            drop_last=False,
            generator=generator,
            worker_init_fn=None
        )
    return loader