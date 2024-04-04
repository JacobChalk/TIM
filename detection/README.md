# TIM: A Time Interval Machine for Audio-Visual Action Recognition - Detection Variant

## Requirements

The requirements for TIM can be found in the main README of this repository. For detection, in addition to the requirements, you must set up the C++ NMS using `python setup.py install --user`.

Be sure to export the relevant directories to your Python Path, e.g. `export PYTHONPATH=/path/to/TIM/detection/time_interval_machine:path/to/TIM/detection/scripts:$PYTHONPATH`.

**NOTE:** This code uses code from the [ActionFormer GitHub](https://github.com/happyharrycn/actionformer_release).

## Features

The features used for this project can be extracted by following the instructions in the `feature_extractors` folder.

## Pretrained models

We provide the pretrained detection models for TIM [here](). This file contains a subfolder with pretrained models for all detection tasks: EPIC-100-Verb, EPIC-100-Noun, Perception Test Action and Perception Test Sound.

## Ground Truth

The Ground Truth files for TIM can be found in the main README of this repository.

## Trainig TIM for Detection

### EPIC-KITCHENS-100:
To train TIM on EPIC-KITCHENS-100 for verbs, run:

```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/epic_visual_features \
--video_train_action_pickle /path/to/epic_100_train_annotations \
--video_train_context_pickle /path/to/epic_train_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/to/epic_audio_features \
--audio_train_context_pickle /path/to/epic_train_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/epic_kitchens_video_metadata \
--verb_only True
```

The noun version of TIM can be run by running the above and changing `--verb_only False`.

### Perception Test Action & Perception Test Sound
To train TIM on Perception Test Action, run:

```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/perception_test_visual_features \
--video_train_action_pickle /path/to/perception_test_action_train_annotations \
--video_train_context_pickle /path/to/perception_test_train_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/perception_test_audio_features \
--audio_train_context_pickle /path/to/perception_test_train_audio_feature_intervals \
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
--video_train_context_pickle /path/to/perception_test_train_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/perception_test_audio_features \
--audio_train_action_pickle /path/to/perception_test_sound_train_annotations \
--audio_train_context_pickle /path/to/perception_test_train_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/perception_test_video_metadata \
--data_modality 'audio' \
--feat_stride 2
```

## Validation

To validate TIM on EPIC-100 verbs, run:

```[bash]
python scripts/run_net.py \
--extract_feats \
--output_dir /path/to/output \
--video_data_path /path/to/epic_visual_features \
--video_val_action_pickle /path/to/epic_100_val_annotations \
--video_val_context_pickle /path/to/epic_val_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/to/epic_audio_features \
--audio_val_context_pickle /path/to/epic_val_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/epic_kitchens_video_metadata \
--verb_only True
```

This will extract dense predictions across the validation set and save it to `/path/to/output/features/EPIC_100_validation.pth.tar`. You can then evaluate the extract predictions. First, change directory to `eval_detection`. To evaluate EPIC-100, run:

```
python format_prediction_epic.py /path/to/output/predictions /path/to/groud/truth/annotations --task <task-of-model>
```

Where `<task-of-model>` refers to `verb` or `noun`.

To evaluate Perception Test Action, or Perception Test Sound run

```[bash]
python scripts/run_net.py \
--extract_feats \
--output_dir /path/to/output \
--video_data_path /path/to/perception_test_visual_features \
--video_val_action_pickle /path/to/perception_test_action_val_annotations \
--video_val_context_pickle /path/to/perception_test_val_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/to/perception_test_audio_features \
--audio_val_context_pickle /path/to/perception_test_val_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/perception_test_video_metadata \
--data_modality 'audio' # Use only if validating Perception Test Sound
```

To extract dense predictions, then run:

```
python format_prediction.py /path/to/output/predictions /path/to/groud/truth/annotations
```

**NOTE:** These scripts will run the evaluation scripts using `subprocess`, so it is important to run while in the `eval_detection` folder.

## License

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
