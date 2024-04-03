# TIM: A Time Interval Machine for Audio-Visual Action Recognition

## Requirements

The requirements for TIM can be found in the main README of this repository.

Be sure to export the relevant directories to your Python Path, e.g. `export PYTHONPATH=/path/to/TIM/recognition/time_interval_machine:path/to/TIM/recognition/scripts:$PYTHONPATH`.

## Features

The features used for this project can be extracted by following the instructions in the `feature_extractors` folder.

## Pretrained models

We provide the pretrained recognition models for TIM here:

[EPIC-100]()

[Perception Test]()

[AVE]()

## Ground-Truth

We provide the necessary ground-truth files for all datasets here:

[Train-Split]()

**NOTE:** These annotation files have been cleaned to be compatible with the TIM codebase

## Trainig TIM for Recognition

### EPIC-KITCHENS-100 & EPIC-Sounds
To train TIM jointly on EPIC-KITCHENS-100 and EPIC-Sounds, run:

```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/epic_visual_features \
--video_train_action_pickle /path/to/epic_100_train_annotations \
--video_val_action_pickle /path/to/epic_100_validation_annotations \
--video_train_context_pickle /path/to/epic_100_train_visual_feature_intervals \
--video_val_context_pickle /path/to/epic_100_validation_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/to/epic_audio_features \
--audio_train_action_pickle /path/to/epic_sounds_train_annotations \
--audio_val_action_pickle /path/to/epic_sounds_validation_annotations \
--audio_train_context_pickle /path/to/epic_sounds_train_audio_feature_intervals \
--audio_val_context_pickle /path/to/epic_sounds_validation_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/epic_kitchens_video_metadata \
--lambda_audio 0.01
```

**NOTE:** Either dataset can be trained individually by changing the `data_modality` flag to either `audio` or `visual`. To use only audio or visual features change the `model_modality` flag to `audio` or `visual`.

### Perception Test Action & Perception Test Sound
To train TIM jointly on Perception Test Action & Perception Test Sound, run:

```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/perception_test_visual_features \
--video_train_action_pickle /path/to/perception_test_action_train_annotations \
--video_val_action_pickle /path/to/perception_test_action_validation_annotations \
--video_train_context_pickle /path/to/perception_test_action_train_visual_feature_intervals \
--video_val_context_pickle /path/to/perception_test_action_validation_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/perception_test_audio_features \
--audio_train_action_pickle /path/to/perception_test_sound_train_annotations \
--audio_val_action_pickle /path/to/perception_test_sound_validation_annotations \
--audio_train_context_pickle /path/to/perception_test_sound_train_audio_feature_intervals \
--audio_val_context_pickle /path/to/perception_test_sound_validation_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/perception_test_video_metadata \
--feat_stride 2 \
--feat_dropout 0.1 \
--seq_dropout 0.1 \
--include_verb_noun False
```

### AVE
To train TIM on AVE, run:
```

```

## Validation

### EPIC-KITCHENS-100 & EPIC-Sounds

### Perception Test Action & Perception Test Sound

### AVE


## License

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
