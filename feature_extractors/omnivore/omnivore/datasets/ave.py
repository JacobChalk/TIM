import os
import pandas as pd
import torch
import torch.utils.data
import cv2
import numpy as np
import random
from torchvision import transforms

import omnivore.utils.logging as logging

from .build import DATASET_REGISTRY
from .ave_record import AVEVideoRecord

from . import autoaugment as autoaugment
from . import transform as transform
from . import utils as utils
from .frame_loader import pack_frames_to_video_clip

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ave(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        self.cfg = cfg 
        self.mode = mode
        self._num_clips = (
            cfg.TEST.NUM_FEATURES * cfg.TEST.NUM_SPATIAL_CROPS
        )

        logger.info("Constructing AVE {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_annotations_pickle = [self.cfg.AVE.TEST_LIST]
        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self._video_records = []
        self._spatial_temporal_idx = []

        for file in path_annotations_pickle:
            for ii, tup in enumerate(pd.read_pickle(file).iterrows()):
                for idx in range(self._num_clips):
                    record = AVEVideoRecord(tup)
                    
                    # For running on whole dataset
                    self._video_records.append(record)
                    self._spatial_temporal_idx.append(idx)
                
        assert (
                len(self._video_records) > 0
        ), "Failed to load AVE split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing AVE dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            assert len({min_scale, max_scale, crop_size}) == 1
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        frames = pack_frames_to_video_clip(self.cfg, self._video_records[index], self.cfg.TEST.DATASET)

        if self.cfg.DATA.USE_RAND_AUGMENT and (temporal_sample_index != 0):
            # Transform to PIL Image
            frames = [transforms.ToPILImage()(frame.squeeze().numpy()) for frame in frames]

            # Perform RandAugment
            img_size_min = crop_size
            auto_augment_desc = "rand-m15-mstd0.5-inc1"
            aa_params = dict(
                translate_const=int(img_size_min * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in self.cfg.DATA.MEAN]),
            )
            seed = random.randint(0, 100000000)
            frames = [autoaugment.rand_augment_transform(
                auto_augment_desc, aa_params, seed)(frame) for frame in frames]

            # To Tensor: T H W C
            frames = [torch.tensor(np.array(frame)) for frame in frames]
            frames = torch.stack(frames)
            
        # For omnivore style frame transform
        if frames.shape[1] < frames.shape[2]:
            scale = min_scale / frames.shape[1]
        else:
            scale = min_scale / frames.shape[2]

        frames = [
                cv2.resize(
                    img_array.numpy(),
                    (0,0),
                    fx=scale,fy=scale,  # The input order for OpenCV is w, h.
                )
                for img_array in frames
        ]
        frames = np.concatenate(
            [np.expand_dims(img_array, axis=0) for img_array in frames],
            axis=0,
        )
        frames = torch.from_numpy(np.ascontiguousarray(frames))
        frames = torch.flip(frames,dims=[3]) # from bgr to rgb
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )
        label = self._video_records[index].label
        metadata = self._video_records[index].metadata
        if frames.shape[-1] == 49:
            print(self._video_records.untrimmed_video_name)

        return frames, label, index, metadata


    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            #assert len({min_scale, max_scale, crop_size}) == 1
            #frames, _ = transform.random_short_side_scale_jitter(
            #    frames, min_scale, max_scale
            #)
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
