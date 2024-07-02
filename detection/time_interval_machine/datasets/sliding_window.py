import pandas as pd
import numpy as np
import torch
import math
import os

from torch.utils import data

import time_interval_machine.utils.logging as logging


logger = logging.get_logger(__name__)

def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = hours * 3600.00 + minutes * 60.0 + seconds
    return total_seconds

def load_feats(feat_info, data_path, mode):
    feat_times = {}
    feats = {}

    v_ids = feat_info['video_id'].unique().tolist()
    for v_id in v_ids:
        vid_times = feat_info[feat_info['video_id'] == v_id].sort_values('start_sec')
        vid_times = vid_times.drop(columns=['video_id', 'narration_sec'])
        feat_times[v_id] = torch.from_numpy(vid_times.to_numpy()).float()

        vid_feats = np.load(os.path.join(data_path, f'{mode}/{v_id}.npy'))
        feats[v_id] = torch.from_numpy(vid_feats)

    return feat_times, feats

class SlidingWindowDataset(data.Dataset):
    def __init__(self,
                v_data_path,
                a_data_path,
                v_action_labels_pickle,
                a_action_labels_pickle,
                v_context_labels_pickle,
                a_context_labels_pickle,
                video_info_pkl,
                v_feature_dim=1024,
                a_feature_dim=2304,
                num_feats=50,
                feat_stride=2,
                feat_gap=0.2,
                window_stride=1.0,
                mode='train',
                data_modality='audio_visual',
                model_modality='audio_visual',
                dataset_name='epic',
                get_gt_segments=True,
                include_verb_noun=False,
                verb_only=True
            ):

        logger.info("Constructing dataset for split : {}".format(mode))

        # Initialise parameters
        self.dataset_name = dataset_name
        self.mode = mode
        self.data_modality = data_modality
        self.model_modality = model_modality

        self.v_feature_dim = v_feature_dim
        self.a_feature_dim = a_feature_dim

        self.num_feats = num_feats
        self.feat_stride = feat_stride
        self.feat_gap = feat_gap
        self.window_size = self.num_feats * feat_gap * feat_stride
        self.window_stride = window_stride
        self.v_num_aug = 1
        self.a_num_aug = 1
        self.max_visual_actions = 0
        self.max_audio_actions = 0
        self.num_actions = 0
        self.min_query = 2*self.window_size
        self.max_query = 0
        self.avg_query = 0
        self.get_gt_segments = get_gt_segments
        self.include_verb_noun = include_verb_noun
        self.verb_only = verb_only


        logger.info("Caching Features")
        self.cache_features(
                            v_data_path,
                            a_data_path,
                            v_context_labels_pickle,
                            a_context_labels_pickle
                        )

        logger.info("Creating Windows")
        self.init_windows(
                            v_action_labels_pickle,
                            a_action_labels_pickle,
                            video_info_pkl
                        )

        self.max_window_actions = self.max_visual_actions + self.max_audio_actions
        out_string = (f'{mode.capitalize()} Sliding Window dataset constructed. '\
                    f'Number of {self.window_size} Second Windows: {len(self.windows)}')
        if get_gt_segments:
            out_string += (f'\n\t\t\t\t\tTotal Actions: {self.num_actions}\n' \
                        f'\t\t\t\t\tMax actions in window: {self.max_window_actions}\n' \
                        f'\t\t\t\t\tMin Action Size: {self.min_query}\n' \
                        f'\t\t\t\t\tMax Action Size: {self.max_query}')

        logger.info(out_string)

    def cache_features(
                    self,
                    v_data_path,
                    a_data_path,
                    v_feat_times_pkl,
                    a_feat_times_pkl
                ):

        if "visual" in self.model_modality:
            logger.info("Loading visual data")
            feat_times = pd.read_pickle(v_feat_times_pkl)
            v_info = load_feats(feat_times, v_data_path, self.mode)
            self.v_feat_times, self.v_feats = v_info
            self.v_num_aug = list(self.v_feats.values())[0].size(1)
        else:
            self.v_feat_times = None
            self.v_feats = None
            self.v_num_aug = 0

        if "audio" in self.model_modality:
            logger.info("Loading audio data")
            feat_times = pd.read_pickle(a_feat_times_pkl)
            a_info = load_feats(feat_times, a_data_path, self.mode)
            self.a_feat_times, self.a_feats = a_info
            self.a_num_aug = list(self.a_feats.values())[0].size(1)
        else:
            self.a_feat_times = None
            self.a_feats = None
            self.a_num_aug = 0

    def init_windows(
                    self,
                    v_labels_pkl,
                    a_labels_pkl,
                    video_info_pkl
                ):
        keep_columns = [
                "video_id",
                "start_sec",
                "stop_sec",
                "verb_class",
                "noun_class",
                "action_class",
                "class_id"
            ]

        # Format visual and audio ground truth dataframes to a matching format
        if "visual" in self.data_modality:
            v_actions = pd.read_pickle(v_labels_pkl)
            v_actions["start_sec"] = v_actions["start_timestamp"].apply(timestamp_to_seconds)
            v_actions["stop_sec"] = v_actions["stop_timestamp"].apply(timestamp_to_seconds)
            v_actions["class_id"] = [-1] * v_actions.shape[0]
            if "verb_class" not in v_actions.columns:
                v_actions["verb_class"] = [-1] * v_actions.shape[0]
                v_actions["noun_class"] = [-1] * v_actions.shape[0]

            v_actions = v_actions[keep_columns]
            v_actions.index = v_actions.index.set_names(['narration_id'])
            v_actions['duration'] = v_actions['stop_sec'] - v_actions['start_sec']
            v_actions = v_actions[v_actions['duration'] < self.window_size]
            v_actions = v_actions.reset_index()
            v_actions['narration_id'] = v_actions['video_id'].apply(lambda x: f"v_{x}")

        if "audio" in self.data_modality:
            a_actions = pd.read_pickle(a_labels_pkl)
            a_actions["start_sec"] = a_actions["start_timestamp"].apply(timestamp_to_seconds)
            a_actions["stop_sec"] = a_actions["stop_timestamp"].apply(timestamp_to_seconds)
            a_actions["verb_class"] = [-1] * a_actions.shape[0]
            a_actions["noun_class"] = [-1] * a_actions.shape[0]
            a_actions["action_class"] = [-1] * a_actions.shape[0]

            a_actions = a_actions[keep_columns]
            a_actions.index = a_actions.index.set_names(['narration_id'])
            a_actions['duration'] = a_actions['stop_sec'] - a_actions['start_sec']
            a_actions = a_actions[a_actions['duration'] < self.window_size]
            a_actions = a_actions.reset_index()
            a_actions['narration_id'] = a_actions['video_id'].apply(lambda x: f"a_{x}")


        if self.data_modality == "visual":
            actions = v_actions
        elif self.data_modality == "audio":
            actions = a_actions
        else:
            actions = pd.concat([v_actions, a_actions], axis=0)
            actions = actions.reset_index(drop=True)

        self.num_actions = actions.shape[0]
        video_info = pd.read_pickle(video_info_pkl)
        video_info = video_info[video_info.index.isin(actions['video_id'].unique())]
        actions = actions.groupby('video_id')

        # Create Windows
        self.windows = []
        windows_path = self.create_windows_path(v_labels_pkl, a_labels_pkl)

        # Check if windows are precomputed
        if not os.path.exists(windows_path):
            for vid, data in video_info.iterrows():
                video_duration = math.ceil(data['duration'])
                vid_feat_times = self.v_feat_times[vid] if 'visual' in self.model_modality else self.a_feat_times[vid]
                num_windows_in_vid = max(math.ceil((math.ceil(video_duration) - self.window_size) / self.window_stride) + 1, 1)
                vid_actions = actions.get_group(vid).copy()

                # Some labels are longer than video info duration
                vid_actions['stop_sec'] = vid_actions['stop_sec'].apply(lambda x: min(x, video_duration))

                for w in range(num_windows_in_vid):
                    win_start = self.window_stride * w
                    win_stop = min(video_duration, win_start + self.window_size)
                    win_feats = self.get_window_features(vid_feat_times, win_start, win_stop)

                    window_info = {
                        'video_id': vid,
                        'start_sec': win_start,
                        'stop_sec': win_stop,
                        'feat_indices': win_feats
                    }

                    if self.get_gt_segments:
                        actions_in_window = vid_actions[
                                    (vid_actions['start_sec'] >= win_start)
                                    & (vid_actions['stop_sec'] <= win_stop)
                                ].copy()

                        if actions_in_window.shape[0] > 0:
                            self.min_query = min(self.min_query, min(actions_in_window['duration'].tolist()))
                            self.max_query = max(self.max_query, max(actions_in_window['duration'].tolist()))
                            actions_in_window = actions_in_window.drop(columns=['video_id', 'duration'])

                            action_times = np.array(actions_in_window[['start_sec', 'stop_sec']])
                            action_times = torch.from_numpy(action_times).float()

                            action_labels = np.array(actions_in_window[['verb_class', 'noun_class', 'action_class', 'class_id']])
                            action_labels = torch.from_numpy(action_labels).long()

                            n_ids = actions_in_window['narration_id'].tolist()

                            visual_actions = [i for (i, n_id) in enumerate(n_ids) if 'v_' in n_id]
                            audio_actions = [i for (i, n_id) in enumerate(n_ids) if 'a_' in n_id]

                            if len(visual_actions) > self.max_visual_actions:
                                self.max_visual_actions = actions_in_window.shape[0]

                            if len(audio_actions) > self.max_audio_actions:
                                self.max_audio_actions = actions_in_window.shape[0]

                            window_info.update({
                                    'v_gt_segments': action_times[visual_actions],
                                    'a_gt_segments': action_times[audio_actions],
                                    'v_labels': action_labels[visual_actions],
                                    'a_labels': action_labels[audio_actions]
                                })
                        else:
                            window_info.update({
                                'v_gt_segments': torch.empty((0, 2)).float(),
                                'a_gt_segments': torch.empty((0, 2)).float(),
                                'v_labels': torch.empty((0, 4)).long(),
                                'a_labels': torch.empty((0, 4)).long()
                            })

                        self.windows.append(window_info)
                    else:
                        window_info.update({
                            'v_gt_segments': torch.empty((0, 2)).float(),
                            'a_gt_segments': torch.empty((0, 2)).float(),
                            'v_labels': torch.empty((0, 4)).long(),
                            'a_labels': torch.empty((0, 4)).long()
                        })
                        self.windows.append(window_info)


            all_windows = {
                            "windows": self.windows,
                            "max_vis": self.max_visual_actions,
                            "max_aud": self.max_audio_actions,
                            "min_query": self.min_query,
                            "max_query": self.max_query
                        }
            torch.save(all_windows, windows_path)
        else:
            logger.info(f"Loading precomputed windows from {windows_path}")
            all_windows = torch.load(windows_path)
            self.windows = all_windows['windows']
            self.max_visual_actions = all_windows['max_vis']
            self.max_audio_actions = all_windows['max_aud']
            self.min_query = all_windows['min_query']
            self.max_query = all_windows['max_query']

        self.min_query = round(self.min_query, 3)
        self.max_query = round(self.max_query, 3)

    def create_windows_path(self, v_labels_pkl, a_labels_pkl):
        windows_path = "precomputed_windows/"

        if "visual" in self.data_modality:
            if self.dataset_name not in str(v_labels_pkl).lower():
                windows_path += f"{self.dataset_name.upper()}_"
            v_label_name = str(v_labels_pkl).split('/')[-1].replace('.pkl', '')
            windows_path += f"{v_label_name}_"

        if "audio" in self.data_modality:
            if self.dataset_name not in str(a_labels_pkl).lower() and self.data_modality != "audio_visual":
                windows_path += f"{self.dataset_name.upper()}"
            a_labels_name = str(a_labels_pkl).split('/')[-1].replace('.pkl', '')
            windows_path += f"{a_labels_name}_"
        hop_secs = round(self.feat_stride * self.feat_gap, 3)
        windows_path += f"win_{self.num_feats}_{hop_secs}_{self.window_size}_{self.window_stride}.pth"
        return windows_path


    def __getitem__(self, index):
        window = self.windows[index]
        video_id = window['video_id']
        feat_indices = window['feat_indices']
        v_gt_segments = torch.round(window['v_gt_segments'] - window['start_sec'], decimals=3)
        a_gt_segments = torch.round(window['a_gt_segments'] - window['start_sec'], decimals=3)
        v_labels = window['v_labels']
        a_labels = window['a_labels']

        # Collect Times and Input Data
        times = torch.zeros(size=(0, 2))
        v_data = torch.empty(size=(0,))
        a_data = torch.empty(size=(0,))

        if "visual" in self.model_modality:
            v_aug_indices = torch.randint(
                                    low=0,
                                    high=self.v_num_aug,
                                    size=(self.num_feats,),
                                    dtype=torch.long
                                )
            v_data = self.v_feats[video_id][feat_indices, v_aug_indices]
            v_input_feat_times = self.v_feat_times[video_id][feat_indices, :2]
            times = torch.cat([times, v_input_feat_times], dim=0)

        if "audio" in self.model_modality:
            a_aug_indices = torch.randint(
                                    low=0,
                                    high=self.a_num_aug,
                                    size=(self.num_feats,),
                                    dtype=torch.long
                                )
            a_data = self.a_feats[video_id][feat_indices, a_aug_indices]
            a_input_feat_times = self.a_feat_times[video_id][feat_indices, :2]
            times = torch.cat([times, a_input_feat_times], dim=0)

        # Normalize times and make relative to input size
        times = torch.round((times - window['start_sec']), decimals=3) / self.window_size
        times = torch.clamp(times, min=0.0)

        # Collect and pad queries+metadata
        v_to_pad = (0, 0 , 0, self.max_visual_actions - v_labels.size(0))
        v_gt_segments = torch.nn.functional.pad(v_gt_segments, v_to_pad, "constant", 0.0)
        v_labels = torch.nn.functional.pad(v_labels, v_to_pad, "constant", -1)

        # Pad queries to max size and form times for Time MLP
        a_to_pad = (0, 0 , 0, self.max_audio_actions - a_labels.size(0))
        a_gt_segments = torch.nn.functional.pad(a_gt_segments, a_to_pad, "constant",  0.0)
        a_labels = torch.nn.functional.pad(a_labels, a_to_pad, "constant", -1)


        if self.dataset_name == 'epic' and not self.include_verb_noun:
            if self.verb_only:
                vis_action_target = v_labels[:, 0]
            else:
                vis_action_target = v_labels[:, 1]
        else:
            vis_action_target = v_labels[:, 2]

        label = {
                'v_gt_segments': torch.clamp(v_gt_segments  / self.window_size, min=0.0),
                'a_gt_segments': torch.clamp(a_gt_segments / self.window_size, min=0.0),
                'verb': v_labels[:, 0],
                'noun': v_labels[:, 1],
                'action': vis_action_target,
                'class_id': a_labels[:, 3]
            }

        metadata = {
            'window_start': window['start_sec'],
            'window_size': self.window_size,
            'video_id': video_id
        }


        return v_data, a_data, times, label, metadata

    def __len__(self):
        return len(self.windows)

    def get_window_features(self, feat_times, window_start, window_stop):
        # Get features to represent windowed input
        start_time = max(0.0, window_start)
        input_start = np.absolute(feat_times[:, 0] - start_time).argmin()
        input_end = np.absolute(feat_times[:, 1] - window_stop).argmin()

        feat_indices = list(range(input_start, input_end, self.feat_stride))
        feat_indices = np.clip(feat_indices, 0, len(feat_times) - 1)
        feat_indices = torch.from_numpy(feat_indices).long()

        num_pad = self.num_feats - feat_indices.size(0)
        final_index = feat_indices[-1]
        feat_indices = torch.nn.functional.pad(feat_indices, (0, num_pad), "constant", final_index)

        return feat_indices
