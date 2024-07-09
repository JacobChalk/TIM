# TIM: A Time Interval Machine for Audio-Visual Action Recognition - Recognition Variant

## Requirements

The requirements and installation instructions for TIM can be found in the main README of this repository.

Be sure to export the relevant directories to your Python Path, e.g. `export PYTHONPATH=/path/to/TIM/recognition/time_interval_machine:path/to/TIM/recognition/scripts:$PYTHONPATH`.

## Features

The features used for this project can be extracted by following the instructions in the `feature_extractors` folder.

## Pretrained models

We provide the pretrained recognition models in the following:

- [EPIC-KITCHENS-100 + EPIC-SOUNDS](https://www.dropbox.com/scl/fi/taqoclgnhjyeoapnb61pq/epic_100_epic_sounds.pth.tar?rlkey=wbfcnlbpdzf3lt35w6ugtc44w&dl=0)
- [EPIC-KITCHENS-100 (Visual-Only)](https://www.dropbox.com/scl/fi/udqes9jcyl1at05xljdgn/epic_visual_only.pth.tar?rlkey=56pe4jjnlyjxg1o2tris4npyi&st=ih7fyqvp&dl=0)
- [Perception Test Action + Sound](https://www.dropbox.com/scl/fi/xzt8rbl19cumgl0v3gl2d/percetion_test_action_sound.pth.tar?rlkey=qsd7vbpddnftpk4mjq4j8dpnm&dl=0)
- [AVE](https://www.dropbox.com/scl/fi/fy3wdorrhfdx9nfilwfem/ave.pth.tar?rlkey=u8nyvs0m11msm5hmfq7rnzs2q&st=l5hrs3zp&dl=0)

## Ground Truth

The Ground Truth files for TIM can be found in the main README of this repository.

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
--audio_data_path /path/to/perception_test_audio_features \
--audio_train_action_pickle /path/to/perception_test_sound_train_annotations \
--audio_val_action_pickle /path/to/perception_test_sound_validation_annotations \
--audio_train_context_pickle /path/to/perception_test_sound_train_audio_feature_intervals \
--audio_val_context_pickle /path/to/perception_test_sound_validation_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/perception_test_video_metadata \
--dataset perception \
--feat_stride 2 \
--feat_dropout 0.1 \
--seq_dropout 0.1 \
--include_verb_noun False
```

### AVE

To train TIM on AVE with vgg features, run:

```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/AVE_visual_features \
--video_train_action_pickle /path/to/AVE_train_annotations \
--video_val_action_pickle /path/to/AVE_validation_annotations \
--video_train_context_pickle /path/to/AVE_visual_feature_intervals \
--video_val_context_pickle /path/to/AVE_validation_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/to/AVE_audio_features \
--audio_train_action_pickle /path/to/AVE_train_annotations \
--audio_val_action_pickle /path/to/AVE_validation_annotations \
--audio_train_context_pickle /path/to/AVE_train_audio_feature_intervals \
--audio_val_context_pickle /path/to/AVE_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/AVE_video_metadata \
--dataset ave \
--feat_stride 1 \
--feat_gap 1.0 \
--num_feats 10 \
--feat_dropout 0.1 \
--seq_dropout 0.1 \
--d_model 256 \
--apply_feature_pooling True \
--lr 5e-4 \
--lambda_audio 0.1 \
--lambda_drloc 0.1 \
--mixup_alpha 0.5 \
--include_verb_noun False
```

To train TIM with Omnivore + AuditorySlowFast features, run
```[bash]
python scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/AVE_visual_features \
--video_train_action_pickle /path/to/AVE_train_annotations \
--video_val_action_pickle /path/to/AVE_validation_annotations \
--video_train_context_pickle /path/to/AVE_visual_feature_intervals \
--video_val_context_pickle /path/to/AVE_validation_visual_feature_intervals \
--visual_input_dim <channel-size-of-visual-features> \
--audio_data_path /path/to/AVE_audio_features \
--audio_train_action_pickle /path/to/AVE_train_annotations \
--audio_val_action_pickle /path/to/AVE_validation_annotations \
--audio_train_context_pickle /path/to/AVE_train_audio_feature_intervals \
--audio_val_context_pickle /path/to/AVE_audio_feature_intervals \
--audio_input_dim <channel-size-of-audio-features> \
--video_info_pickle /path/to/AVE_video_metadata \
--dataset ave \
--feat_stride 2 \
--num_feats 25 \
--feat_dropout 0.1 \
--d_model 256 \
--lr 5e-4 \
--lambda_drloc 0.1 \
--mixup_alpha 0.5 \
--include_verb_noun False
```

## Validation

You can validate a pretrained version of TIM on all the previous datasets by running the previous commands, but changing the `--train` flag to `--validate` and adding the flag `--pretrained_model /path/to/pretrained_model`.

## Extract Predictions

You can extract predictions from TIM for each ground truth annotation on all the previous datasets by running the previous commands, but changing the `--train` flag to `--extract_feats` and adding the flag `--pretrained_model /path/to/pretrained_model`. These results will be saved to a file `/output_path/features/dataset_name_dataset_split.pkl`. This will be a dictionary containing:

```[python]
{
    "action": The predicted visual actions of shape (N_vids, N_vis_classes),
    "audio": The predicted audio actions of shape (N_vids, N_audio_classes),
    "v_narration_ids": The unique ids of each visual ground truth segment,
    "a_narration_ids": The unique ids of each audio ground truth segment,
    "verb": The predictied verb classes of shape (N_vids, 97) (EPIC Only),
    "noun": The predictied noun classes of shape (N_vids, 300) (EPIC Only),
}
```

This dictionary can be manipulated into the correct format for sumbission to the [EPIC-Kitchens Action Recognition Challenge](https://github.com/epic-kitchens/C1-Action-Recognition).

## License

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
