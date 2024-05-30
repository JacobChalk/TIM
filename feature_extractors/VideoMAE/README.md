# TIM: A Time Interval Machine for Audio-Visual Action Recognition - VideoMAE Backbone

This section of code is a condensed version of [VideoMAE](https://github.com/MCG-NJU/VideoMAE), used for TIM, from the [InternVideo GitHub](https://github.com/OpenGVLab/InternVideo).

## Requirements

Instructions on how to install the necessary packages can be found on the InternVideo GitHub [here](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo1/Pretrain/VideoMAE/README.md).

## Download the Data
Please refer to the [Omnivore ReadME](https://github.com/JacobChalk/TIM/blob/main/feature_extractors/omnivore/README.md) for information on how to download the data for each dataset.

## Training VideoMAE on EPIC

To replicate our pretrained VideoMAE model on EPIC-KITCHENS-100, you will first need to download the EPIC-100 videos and extract the frames. Instructions can be found [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/README.md#erratum). You will also need to download the annotations given [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations).

You will also need to download the pretrained VideoMAE provided by InternVideo, by following the link [here](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo1/README.md) and navigating to "Pretrained Models -> VideoMAE-L w/ UnlabelledHybrid (1M)".

Once you have done the above, you can train VideoMAE with:

```[bash]
python -u run_class_finetuning.py \
    --model vit_large_patch16_224 \
    --data_set EK100 \
    --nb_classes 97 300 \
    --anno_path /path/to/epic-100-annotations \
    --data_path /path/to/extracted/epic-frames \
    --finetune /path/to/videomae-pretrained-model \
    --log_dir /path/to/log-dir \
    --output_dir /path/to/output-dir \
    --batch_size 2 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 0.0003 \
    --num_workers 6 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --head_drop_rate 0.3 \
    --layer_decay 0.8 \
    --mixup 0.0 \
    --cutmix 0.0 \
    --epochs 50 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed
```

If you are unable to train VideoMAE yourself, we provided our pretrained model [here](https://www.dropbox.com/scl/fi/kr14exj9ipfcoth01thld/videomae_epic.pth.tar?rlkey=d9jqjqp2b3zy1440qcbdog3x4&dl=0).

## Extracting VideoMAE Features EPIC

To extract the VideoMAE Features for EPIC, run:

```[bash]
python -u feature_extraction.py \
--data_path /path/to/extracted/epic-frames \
--save_path /path/to/save/features \
--video_csv /path/to/desired/feature/times \
--ckpt_path /path/to/pretrained/model
```

**NOTE:** VideoMAE-L is very resource intensive. We required x8 Tesla V100 GPUs with 32GB vRAM to pretrain the model and extract features in a timely manner.
