"""Extract features for temporal action detection datasets"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from timm.models import create_model
from torch.utils.data import Dataset
import cv2

import modeling_finetune # Important, needed to initialise models
from torchvision import transforms
import video_transforms as video_transforms
import volume_transforms as volume_transforms

def get_args():
    parser = argparse.ArgumentParser(
        'Extract features using the videomae model', add_help=False)

    parser.add_argument(
        '--video_csv',
        type=str,
        help='CSV containing video info')
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to frames')
    parser.add_argument(
            '--save_path',
            type=str,
            help='Path for saving features'
        )
    parser.add_argument(
        '--dataset',
        default='ek100',
        type=str,
        help='dataset')
    parser.add_argument(
        '--num_classes',
        default=[97, 300],
        type=str,
        help='dataset')
    parser.add_argument(
            '--model',
            default='vit_large_patch16_224',
            type=str,
            metavar='MODEL',
            help='Name of model'
        )
    parser.add_argument(
        '--num_aug',
        default=4,
        type=int,
        help='Number of feature sets to extract, > 1 includes augmented sets.'
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        help='Random seed.'
    )
    parser.add_argument(
        '--ckpt_path',
        help='Load from checkpoint')

    args = parser.parse_args()

    args.num_aug = max(args.num_aug, 1)

    if args.dataset == "ek100":
        args.num_classes = [97, 300]
    elif args.dataset == "perception_test":
        args.num_classes = 67

    return args


class FeatureExtractionDataset(Dataset):
    def __init__(self,
                 df_vid_group,
                 video_path,
                 enable_augmentation):
        self.df_vid_group = df_vid_group
        self.video_path = video_path
        self.enable_augmentation = enable_augmentation
        self.transform = video_transforms.Compose([
                video_transforms.Resize(224,
                                        interpolation='bilinear'),
                video_transforms.CenterCrop(size=(224,
                                                    224)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        row = self.df_vid_group.iloc[index]
        start_frame = row['start_frame']
        stop_frame = row['stop_frame']
        num_frames = stop_frame - start_frame
        data = load_frame(self.video_path, num_frames, start_frame)
        if self.enable_augmentation:
            # RandAugment
            aug_transform = video_transforms.create_random_augment(
                input_size=(224, 224),
                auto_augment='rand-m7-n4-mstd0.5-inc1',
                interpolation='bicubic',
            )
            data = [transforms.ToPILImage()(frame) for frame in data]
            data = aug_transform(data)

        frame_q = self.transform(data)  # torch.Size([3, 16, 224, 224])
        return frame_q

    def __len__(self):
        return self.df_vid_group.shape[0]


def load_frame(sample, num_frames, frame_offset, num_segment=16, filename_tmpl='frame_{:010}.jpg'):
    fname = sample

    # handle temporal segments
    average_duration = num_frames // num_segment
    all_index = []
    all_index = list(
        np.multiply(list(range(num_segment)),
                    average_duration) +
        np.ones(num_segment, dtype=int) *
        (average_duration // 2))

    all_index = list(np.array(all_index))
    imgs = []
    for idx in all_index:
        frame_fname = os.path.join(fname,
                                    filename_tmpl.format(idx + 1 + frame_offset))
        # img_bytes = self.client.get(frame_fname)
        with open(frame_fname, "rb") as fp:
            img_bytes = fp.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)
    buffer = np.array(imgs)
    return buffer


def extract_feature(args):
    # Set random seeds
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    rng_generator = torch.manual_seed(args.seed)

    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # get video path
    df_vid = pd.read_pickle(args.video_csv)

    seen_vids = []
    for video_id in df_vid['video_id'].unique():
        url = os.path.join(args.save_path, video_id + '.npy')
        if os.path.exists(url):
            seen_vids.append(url)

    df_vid = df_vid[~df_vid['video_id'].isin(seen_vids)]

    groups = df_vid.groupby('video_id')
    all_vids = [(id, group) for id, group in groups]
    if 'SLURM_LOCALID' in os.environ:
        rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        rank = 0
        world_size = 1
        
    torch.cuda.set_device(rank)
    rank_vids = [vid for i,vid in enumerate(all_vids) if (i % world_size) == rank]

    # get model & load ckpt
    model = create_model(
        args.model,
        img_size=224,
        pretrained=False,
        num_classes=[97, 300] if args.dataset == "ek100" else 67,
        all_frames=16,
        tubelet_size=2,
        drop_rate=0.0,
        drop_path_rate=0.2,
        attn_drop_rate=0.0,
        head_drop_rate=0.3,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        with_cp=False,
        num_segment=1,
    )


    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()

    # Extract Feature
    for video_id, df_vid_group in rank_vids:
        url = os.path.join(args.save_path, video_id + '.npy')

        video_path = os.path.join(args.data_path, f"{video_id}")

        all_sets = []
        for i in range(args.num_aug):
            dataset = FeatureExtractionDataset(
                                            df_vid_group=df_vid_group,
                                            video_path=video_path,
                                            enable_augmentation=(i > 0)
                                        )
            data_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=32,
                                                    num_workers=2,
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    shuffle=False,
                                                    generator=rng_generator,
                                                )
            feature_list = []
            for input_data in data_loader:
                input_data = input_data.cuda()
                with torch.no_grad():
                    feature = model.forward_features(input_data)
                    feature_list.extend(feature.cpu().numpy())
            all_sets.append(np.vstack(feature_list))
            
        if args.num_aug > 1:
            all_sets = np.stack(all_sets, axis=1)
        else:
            all_sets = np.expand_dims(all_sets[0], axis=1)

        # [N, C]
        np.save(url, all_sets)
        print(f'Save feature on {url}')


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
