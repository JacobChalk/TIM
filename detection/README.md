# TIM: A Time Interval Machine for Audio-Visual Action Recognition - Detection Variant

## Requirements

The requirements for TIM can be found in the main README of this repository. For detection, in addition to the requirements, you must set up the C++ NMS using `python setup.py install --user`.

Be sure to export the relevant directories to your Python Path, e.g. `export PYTHONPATH=/path/to/TIM/recognition/time_interval_machine:path/to/TIM/recognition/scripts:$PYTHONPATH`.

**NOTE:** This code uses code from the [ActionFormer GitHub](https://github.com/happyharrycn/actionformer_release).

## Features

The features used for this project can be extracted by following the instructions in the `feature_extractors` folder.

## Pretrained models

We provide the pretrained detection models for TIM here:

[EPIC-100]()

[Perception Test]()

[AVE]()

## Ground-Truth

We provide the necessary ground-truth files for all datasets here:

[EPIC Ground Truths]()

[Perception Test Ground Truths]()

[AVE Ground Truths]()

Each link contains a zip containing: 
- The training split ground truth
- The validation split ground truth
- The video metadata of the dataset

**NOTE:** These annotation files have been cleaned to be compatible with the TIM codebase.

## Trainig TIM for Detection

### EPIC-KITCHENS-100 & EPIC-Sounds
To train TIM on EPIC-KITCHENS-100 for verbs, run:

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
--audio_train_context_pickle /path/to/epic_sounds_train_audio_feature_intervals \
--audio_val_context_pickle /path/to/epic_sounds_validation_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/epic_kitchens_video_metadata \
--verb_only True
```

The noun version of TIM can be run by running the above and changing `--verb_only False`.


To train TIM on EPIC-Sounds, run:

```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/epic_visual_features \
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
--data_modality 'audio'
```

### Perception Test Action & Perception Test Sound
To train TIM on Perception Test Action, run:

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
--audio_train_context_pickle /path/to/perception_test_sound_train_audio_feature_intervals \
--audio_val_context_pickle /path/to/perception_test_sound_validation_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/perception_test_video_metadata \
--feat_stride 2
```

To train TIM on Perception Test Sound, run:

```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/perception_test_visual_features \
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
--data_modality 'audio' \
--feat_stride 2
```

## Validation

You can validate a pretrained version of TIM on all the previous datasets by running the previous commands, but changing the `--train` flag to `--extract_feats` and adding the flag `--pretrained_model /path/to/pretrained_model`.

This will extract dense predictions across the validation set and save it to `/path/to/output/features/dataset_name_dataset_split.pth.tar`. You can then evaluate the extract predictions. First, change directory to `eval_detection`. To evaluate EPIC-100, run:

```
python format_prediction_epic.py /path/to/output/predictions /path/to/groud/truth/annotations --task <task-of-model>
```

Where `<task-of-model>` refers to `verb` or `noun`.

To evaluate Perception Test Action or Perception Test Sound, run:

```
python format_prediction.py /path/to/output/predictions /path/to/groud/truth/annotations
```

**NOTE:** These scripts will run the evaluation scripts using `subprocess`, so it is important to run while in the `eval_detection` folder.

## License

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
