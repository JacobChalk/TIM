import os
import pandas as pd
import pickle
import torch
import h5py
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY
from .epicsounds_record import EpicSoundsAudioRecord

from .spec_augment import combined_transforms
from . import utils as utils
from .audio_loader_epic import pack_audio_epic

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Epicsounds(torch.utils.data.Dataset):

    def __init__(self, cfg):
        self.cfg = cfg

        self._num_clips = cfg.TEST.NUM_FEATURES

        self.audio_dataset = None
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the audio loader.
        """
        path_annotations_pickle = self.cfg.EPICSOUNDS.TEST_LIST

        assert PathManager.exists(path_annotations_pickle), "{} not found".format(
            path_annotations_pickle
        )

        self._audio_records = []
        self._temporal_idx = []
        for tup in pd.read_pickle(path_annotations_pickle).iterrows():
            for idx in range(self._num_clips):
                record = EpicSoundsAudioRecord(tup, sr=self.cfg.AUDIO_DATA.SAMPLING_RATE)

                self._audio_records.append(record)
                self._temporal_idx.append(idx)
        assert (
                len(self._audio_records) > 0
        ), "Failed to load EPIC-SOUNDS split from {}".format(
            path_annotations_pickle
        )
        logger.info(
            "Constructing epicsounds dataloader (size: {}) from {}".format(
                len(self._audio_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the audio index, return the spectrogram, label, audio
        index, and metadata.
        Args:
            index (int): the audio index provided by the pytorch sampler.
        Returns:
            spectrogram (tensor): the spectrogram sampled from the audio. The dimension
                is `channel` x `num frames` x `num frequencies`.
            label (int): the label of the current audio.
            index (int): Return the index of the audio.
        """
        if self.audio_dataset is None:
            self.audio_dataset = h5py.File(self.cfg.EPICSOUNDS.AUDIO_DATA_FILE, 'r')

        temporal_sample_index = self._temporal_idx[index]


        spectrogram = pack_audio_epic(self.cfg, self.audio_dataset, self._audio_records[index], temporal_sample_index)
        # Normalization.
        spectrogram = spectrogram.float()
        if temporal_sample_index != 0:
            # Data augmentation.
            # C T F -> C F T
            spectrogram = spectrogram.permute(0, 2, 1)
            # SpecAugment
            spectrogram = combined_transforms(spectrogram)
            # C F T -> C T F
            spectrogram = spectrogram.permute(0, 2, 1)
        label = self._audio_records[index].label
        spectrogram = utils.pack_pathway_output(self.cfg, spectrogram)
        metadata = self._audio_records[index].metadata
        return spectrogram, label, index, metadata

    def __len__(self):
        return len(self._audio_records)
