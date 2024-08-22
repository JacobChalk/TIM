# TIM: A Time Interval Machine for Audio-Visual Action Recognition - Detection Variant

## Replicating Paper Results
A previous error was discovered in the codebase, fixed by [this commit](https://github.com/JacobChalk/TIM/commit/7b69d3a8a89420db19805f5d1a1f2b0181aca384), where the IOU threshold was never passed to the model while being built, meaning the threshold for the model always adopted the default value of 0.25 as seen in [this line](https://github.com/JacobChalk/TIM/blob/b8e359ec9177d9e7d3b5eefb91ae91d1b7fab3d6/detection/time_interval_machine/models/tim.py#L33). This parameter in the model is responsible for labelling positive/negative proposals based on IOU within the model itself. In addition to this, in the training script we weighted the classification loss of positive proposals by IOU and set a weight of 1.0 for negative queries. This was done by [setting IOU values less than the passed argument as to equal 1.0](https://github.com/JacobChalk/TIM/blob/b8e359ec9177d9e7d3b5eefb91ae91d1b7fab3d6/detection/scripts/train.py#L230). 

However, as the IOU threshold was never given to the model, the loss weighting and the model had different ideas of the IOU threshold. The model always saw a threshold of 0.25, thus labelling queries between 0.25 and 0.6 as positive, whereas the weighting assumed the passed args threshold of 0.6. This means positive queries which had 0.25 <= IOU < 0.6 had a weight of 1.0, but positive queries >= 0.6 had a weight equal to the IOU.

This is an unintended and rather adhoc solution, but it was missed at the time of release. Hence, to replicate our detection results from the paper by training from scratch, you would need to use [this commit](https://github.com/JacobChalk/TIM/tree/b8e359ec9177d9e7d3b5eefb91ae91d1b7fab3d6) of TIM. Our pretrained weights still replicate results in the fixed codebase.

## Requirements

The requirements and installation instructions for TIM can be found in the main README of this repository. For detection, in addition to the requirements, you must set up the C++ NMS using `python setup.py install --user`.

Be sure to export the relevant directories to your Python Path, e.g. `export PYTHONPATH=/path/to/TIM/detection/time_interval_machine:path/to/TIM/detection/scripts:$PYTHONPATH`.

**NOTE:** This code uses code from the [ActionFormer GitHub](https://github.com/happyharrycn/actionformer_release).

## Features

The features used for this project can be extracted by following the instructions in the `feature_extractors` folder.

## Pretrained models

We provide the pretrained detection models in the following:

- [EPIC-KITCHENS-100 Verb](https://www.dropbox.com/scl/fi/tstv5yps3qznfyqthowl4/epic_100_verb.pth.tar?rlkey=blzpf62l6xjt3aefzaj6ruie3&dl=0)
- [EPIC-KITCHENS-100 Noun](https://www.dropbox.com/scl/fi/lyafhr1zn692k4ol66xjr/epic_100_noun.pth.tar?rlkey=y4urlvtqyagwskkijxig7mehh&dl=0)
- [EPIC-Sounds](https://www.dropbox.com/scl/fi/s11osizv5m3synp1aodfg/epic_sounds.pth.tar?rlkey=4nk5rc9saetfcs0kc5b25li9t&dl=0)
- [Perception Test Action](https://www.dropbox.com/scl/fi/jzucxr64s9970bgb78n9n/perception_test_action.pth.tar?rlkey=pqi8n2khj222eu1j5p8c3c2nj&dl=0)
- [Perception Test Sound](https://www.dropbox.com/scl/fi/80fx6uz30dn9owyntdnt9/perception_test_sound.pth.tar?rlkey=4nytiwvad9nmeyrl3ng6a8nd2&dl=0)

## Ground Truth

The Ground Truth files for TIM can be found in the main README of this repository.

## Trainig TIM for Detection

### EPIC-KITCHENS-100

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

To train TIM on EPIC-SOUNDS, run:

```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/epic_visual_features \
--video_train_context_pickle /path/to/epic_train_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/to/epic_audio_features \
--audio_train_action_pickle /path/to/epic_sounds_train_annotations \
--audio_train_context_pickle /path/to/epic_train_audio_feature_intervals \
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
--verb_only True \
--pretrained_model /path/to/pretrained_model
```

This will extract dense predictions across the validation set and save it to `/path/to/output/features/EPIC_100_validation.pth.tar`. You can then evaluate the extract predictions. First, change directory to `eval_detection`. To evaluate EPIC-100, run:

```[bash]
python format_prediction_epic.py \
/path/to/output/features/EPIC_100_validation.pth.tar \
/path/to/groud/truth/annotations \
--task <task-of-model>
```

Where `<task-of-model>` refers to `verb` or `noun`.

To validate TIM on EPIC-Sounds, run:

```[bash]
python scripts/run_net.py \
--extract_feats \
--output_dir /path/to/output \
--video_data_path /path/to/epic_visual_features \
--video_val_context_pickle /path/to/epic_val_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/to/epic_audio_features \
--audio_val_action_pickle /path/to/epic_sounds_val_annotations \
--audio_val_context_pickle /path/to/epic_val_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/epic_kitchens_video_metadata \
--data_modality 'audio' \
--pretrained_model /path/to/pretrained_model
```

This will extract dense predictions across the validation set and save it to `/path/to/output/features/EPIC_Sounds_validation.pth.tar`. You can then evaluate the extract predictions. First, change directory to `eval_detection`. To evaluate EPIC-Sounds, run:

```[bash]
python format_prediction.py \
/path/to/output/features/EPIC_SOUNDS_validation.pth.tar \
/path/to/groud/truth/annotations\
--score_threshold 0.03 \
--sigma 0.25 \
--is_audio
```

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
--data_modality 'audio' # Use only if validating Perception Test Sound \
--pretrained_model /path/to/pretrained_model
```

To extract dense predictions, then run:

```[bash]
python format_prediction.py \
/path/to/output/predictions \
/path/to/groud/truth/annotations \
--is_audio # Use for Perception Test Sound only
```

**NOTE:** These scripts will run the evaluation scripts using `subprocess`, so it is important to run while in the `eval_detection` folder.

## License

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
