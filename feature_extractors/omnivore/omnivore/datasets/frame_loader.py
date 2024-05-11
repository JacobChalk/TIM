import os
import sys
import torch
import numpy as np
from . import utils as utils
from .decoder import get_start_end_idx
import glob

def temporal_sampling(num_frames, start_idx, end_idx, num_samples, start_frame=0, video_last_frame=float('inf')):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index += start_frame
    end_frame = start_frame + num_frames - 1
    index = torch.clamp(index, start_frame, end_frame if end_frame < video_last_frame else video_last_frame).long()
    return index


def pack_frames_to_video_clip(cfg, video_record, dataset='epickitchens'):
    # Load video by loading its extracted frames
    if dataset == 'epickitchens':
        path_to_video = '{}/{}/rgb_frames/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
                                                    video_record.participant,
                                                    video_record.untrimmed_video_name)
        num_frames_video = len(glob.glob('{}/*.jpg'.format(path_to_video)))
        img_tmpl = "frame_{:010d}.jpg"
    elif dataset == 'ave':
        path_to_video = '{}/{}'.format(cfg.AVE.VISUAL_DATA_DIR, video_record.untrimmed_video_name)
        num_frames_video = len(glob.glob('{}/*.jpg'.format(path_to_video)))
        img_tmpl = "frame_{:010d}.jpg"
    elif dataset == 'perception':
        path_to_video = '{}/{}'.format(cfg.PERCEPTION.VISUAL_DATA_DIR, video_record.untrimmed_video_name)
        num_frames_video = len(glob.glob('{}/*.jpg'.format(path_to_video)))
        img_tmpl = "frame_{:010d}.jpg"
    else:
        print("Dataset not implemented : ", dataset)
        sys.exit(1)

    if cfg.DATA.FRAME_SAMPLING == 'like omnivore':

        seg_size = float(video_record.num_frames - 1) / cfg.DATA.NUM_FRAMES
        seq = []
        for i in range(cfg.DATA.NUM_FRAMES):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            seq.append((start + end) // 2)
        frame_idx = torch.tensor(video_record.start_frame + np.array(seq))
        frame_idx = torch.clamp(frame_idx, 1, num_frames_video)
    else:
        print("Data sampling method not implemented : ", cfg.DATA.FRAME_SAMPLING)
        sys.exit(1)

    img_paths = [os.path.join(path_to_video, img_tmpl.format(idx.item())) for idx in frame_idx]
    frames = utils.retry_load_images(img_paths)
    return frames
