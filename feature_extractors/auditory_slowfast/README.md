# Audio feature extraction

This repo contains the audio feature extraction code for Epic-Sounds, Perception Test and AVE dataset.

## Environment

We recommend you to use conda environment.

```[bash]
conda create -n tim_asf python=3.10
conda activate tim_asf
pip install -r requirements.txt
```

## Pretrained models

We use the Auditory SlowFast model, retrained with input = 1 sec. You can replicate these as follows:

- For EPIC-SOUNDS and Perception Test, we used the training script as described in [EPIC-SOUNDS repository](https://github.com/epic-kitchens/epic-sounds-annotations/tree/main/src) repository.
We change the line 11~12 in [this](https://github.com/epic-kitchens/epic-sounds-annotations/blob/c61444e7abb72589d6d805042fabdaa5bd9ed765/src/configs/EPIC-Sounds/slowfast/SLOWFASTAUDIO_8x8_R50.yaml#L11) with `CLIP_SECS:0.999` and `NUM_FRAMES:200`.
- For AVE, we pretrained the model with VGGSound dataset. We use the training script in [this](https://github.com/ekazakos/auditory-slow-fast) repository.
We fixed the line 11 ~ 13 in [here](https://github.com/ekazakos/auditory-slow-fast/blob/6af3fd7277e278315ebbed1430d682dd64a3a80d/configs/VGG-Sound/SLOWFAST_R50.yaml#L11) with `CLIP_SECS:0.999` and `NUM_FRAMES:200`. Note that the sampling rate of audios in Perception Test and EPIC-SOUNDS dataset we used is 24kHz. The sampling rate of AVE audios is 16kHz.

We also provide these pretrained models directly in the following:

- [EPIC-Sounds](https://www.dropbox.com/scl/fi/dfggu2p5mk4wa653rmgx1/asf_epicsounds.pyth?rlkey=s14j28kz3ovp0fnmcdmjxdqvd&dl=0)
- [VGGSound](https://www.dropbox.com/scl/fi/3chl5lnrk3tju75lw6brh/asf_vggsound.pyth?rlkey=gxxdwuu8cpnwct5l71xe1oeb4&dl=0)

## EPIC-SOUNDS

### Download the EPIC-SOUNDS data

A download script is provided for the videos [here](https://github.com/epic-kitchens/download-scripts-100). You will have to extract the untrimmed audios from these videos. Instructions on how to extract and format the audio into a HDF5 dataset can be found on the [Auditory SlowFast](https://github.com/ekazakos/auditory-slow-fast) GitHub repo. Alternatively, you can email [uob-epic-kitchens@bristol.ac.uk](mailto:uob-epic-kitchens@bristol.ac.uk) for access to an existing HDF5 file.

**Contact:** [uob-epic-kitchens@bristol.ac.uk](mailto:uob-epic-kitchens@bristol.ac.uk)

### Necessary files before extracting the EPIC-SOUNDS features

- hdf5 file you get from the previous section
- checkpoint file for Auditory SlowFast pretrained with EPIC-SOUNDS
- yaml file for model / data configuration
- pickle file that has start and end time stamp for frame-level features

### Run the EPIC-SOUNDS feature extraction

First, correctly configure you python path with `export PYTHONPATH=/path/to/TIM/feature_extractors/auditory_slowfast/slowfast:$PYTHONPATH`, then run the following command:

```[bash]
python tools/run_net.py \
--cfg /path/to/configs/EPIC-SOUNDS/SLOWFAST_R50.yaml \
NUM_GPUS <num_gpus> \
OUTPUT_DIR /path/to/output/dataset_split \
EPICSOUNDS.AUDIO_DATA_FILE /path/to/EPICSOUNDS_audio_files \
EPICSOUNDS.TEST_LIST /path/to/EPICSOUNDS_feature_interval_times \
TEST.CHECKPOINT_FILE_PATH /path/to/pretrained_epic_sounds_auditory_slowfast \
TEST.NUM_FEATURES <num_features>
```

The parameters are explained below:

```[bash]
NUM_GPUS : number of gpus you use for extraction. For audio, 1 is enough since it doesn't take a lot of time.
OUTPUT_DIR : output directory to save the features for dataset split
EPICSOUNDS.AUDIO_DATA_FILE : hdf5 path which contains audio waveforms for EPIC-SOUNDS
EPICSOUNDS.TEST_LIST : pickle file containing time intervals for each feature in the EPIC-SOUNDS dataset split
TEST.CHECKPOINT_FILE_PATH : pretrained checkpoint file path
TEST.NUM_FEATURES : Number of feature sets for each time segment. 1 for validation and test set (unaugmented) and 3 for train set (unaugmented+augmented sets).
```

This script will produce `features.npy` and `metadata.npy` in OUTPUT_DIR. `metadata.npy` is only for debugging. You need to rearrange `features.npy` by video_ids.

**NOTE:** The above command will need to be run **per dataset split** i.e. for train, val and test.

### Post processing EPIC features

You need to post-process the features with numpy files. The directory structure should be something like below:

```[bash]
output_dir
├── train     
    ├──P01_01.npy
    ├──P01_02.npy
    ...    
├── val
    ├──P01_11.npy
    ├──P01_12.npy
    ...                  
├── test   
    ├──P01_101.npy
    ├──P02_106.npy
    ...                 
```

To make this, run the following code. You need to change the `feature_file`, `pickle_file` and `out_dir` on your own.

```python
python utils/make_npyfiles.py --feature_file FEATURE_FILE --pickle_file PICKLE_FILE --out_dir OUT_DIR
```

The parameters are expalined below.
```[bash]
feature_file : 'features.npy' that you generated from the previous step.
pickle_file : The pickle file that contains time intervals with narration_id and video_ids. EPICSOUNDS.TEST_LIST in above example.
out_dir : output directory to save the rearranged features. We recommend to put the split name ('train', 'val' or 'test) at the end. (ex - /path/to/outputs/{train,val,test})
```

## Perception test

### Download the Perception Test data

<!-- You need to visit perception_test official website -->
Visit the official [perception_test website](https://github.com/google-deepmind/perception_test) to download the audio files.

### Necessary files before extracting Perception Test features

- audio files you downloaded from previous step
- checkpoint file for Auditory SlowFast pretrained with EPIC-SOUNDS
- yaml file for model / data configuration
- pickle file that has start and end time stamp for frame-level features

### Run the Perception Test feature extraction

First, correctly configure you python path with `export PYTHONPATH=/path/to/TIM/feature_extractors/auditory_slowfast/slowfast:$PYTHONPATH`, then run the following command:

```[bash]
python tools/run_net.py \
--cfg /path/to/configs/PERCEPTION/SLOWFAST_R50.yaml \
NUM_GPUS <num_gpus> \
OUTPUT_DIR /path/to/output/dataset_split \
PERCEPTION.AUDIO_DATA_DIR /path/to/PERCEPTION_audio_files \
PERCEPTION.TEST_LIST /path/to/PERCEPTION_feature_interval_times \
TEST.CHECKPOINT_FILE_PATH /path/to/pretrained_epic_sounds_auditory_slowfast \
TEST.NUM_FEATURES <num_features>
```

The parameters are explained below:

```[bash]
NUM_GPUS : number of gpus you use for extraction. For audio, 1 is enough since it doesn't take a lot of time.
OUTPUT_DIR : output directory to save the features for dataset split
PERCEPTION.AUDIO_DATA_DIR : directory path which contains audio files for Perception Test
PERCEPTION.TEST_LIST : pickle file containing time intervals for each feature in the Perception Test dataset split
TEST.CHECKPOINT_FILE_PATH : pretrained checkpoint file path
TEST.NUM_FEATURES : Number of feature sets for each time segment. 1 for validation and test set (unaugmented) and 3 for train set (unaugmented+augmented sets).
```

This script will produce `features.npy` and `metadata.npy` in OUTPUT_DIR. `metadata.npy` is only for debugging. You need to rearrange `features.npy` by video_ids.

**NOTE:** The above command will need to be run **per dataset split** i.e. for train, val and test.

### Post processing Perception Test features

You need to post-process the features with numpy files. The directory structure should be something like below:

```[bash]
output_dir
├── train     
    ├──video_10001.npy
    ├──video_10003.npy
    ...    
├── val
    ├──video_10000.npy
    ├──video_10004.npy
    ...                  
```

To make this, run the following code. You need to change the `feature_file`, `pickle_file` and `out_dir` on your own.

```python
python utils/make_npyfiles.py --feature_file FEATURE_FILE --pickle_file PICKLE_FILE --out_dir OUT_DIR
```

The parameters are expalined below.
```[bash]
feature_file : 'features.npy' that you generated from the previous step.
pickle_file : The pickle file that contains time intervals with narration_id and video_ids. PERCEPTION.TEST_LIST in above example.
out_dir : output directory to save the rearranged features. We recommend to put the split name ('train', 'val' or 'test) at the end. (ex - /path/to/outputs/{train,val,test})
```

## AVE

### Download the AVE data

<!-- You need to visit AVE official website -->
Visit the official [AVE website](https://github.com/YapengTian/AVE-ECCV18) to download the video files.
To extract the audio files, please refer to [utils/extract_audio.py](utils/extract_audio.py)

### Necessary files before extracting AVE features

- audio files you extracted from previous step
- checkpoint file for Auditory SlowFast pretrained with VGGSound
- yaml file for model / data configuration
- pickle file that has start and end time stamp for frame-level features

### Run AVE feature extraction

First, correctly configure you python path with `export PYTHONPATH=/path/to/TIM/feature_extractors/auditory_slowfast/slowfast:$PYTHONPATH`, then run the following command:

```[bash]
python tools/run_net.py \
--cfg /path/to/configs/AVE/SLOWFAST_R50.yaml \
NUM_GPUS <num_gpus> \
OUTPUT_DIR /path/to/output/dataset_split \
AVE.AUDIO_DATA_DIR /path/to/AVE_audio_files \
AVE.TEST_LIST /path/to/AVE_feature_interval_times \
TEST.CHECKPOINT_FILE_PATH /path/to/pretrained_auditory_slowfast \
TEST.NUM_FEATURES <num_features>
```

The parameters are explained below:

```[bash]
NUM_GPUS : number of gpus you use for extraction. For audio, 1 is enough since it doesn't take a lot of time.
OUTPUT_DIR : output directory to save the features for dataset split
AVE.AUDIO_DATA_DIR : directory path which contains audio files for AVE
AVE.TEST_LIST : pickle file containing time intervals for each feature in the AVE dataset split
TEST.CHECKPOINT_FILE_PATH : pretrained checkpoint file path
TEST.NUM_FEATURES : Number of feature sets for each time segment. 1 for validation and test set (unaugmented) and 5 for train set (unaugmented+augmented sets).
```

This script will produce `features.npy` and `metadata.npy` in OUTPUT_DIR. `metadata.npy` is only for debugging. You need to rearrange `features.npy` by video_ids.

**NOTE:** The above command will need to be run **per dataset split** i.e. for train, val and test.

### Post processing AVE features

You need to post-process the features with numpy files. The directory structure should be something like below:

```[bash]
output_dir
├── train     
    ├──004KfU7bgyg.npy
    ├──0095-_8T5ZY.npy
    ...    
├── val
    ├──00N83yxKYUI.npy
    ├──01eOqSIF9PE.npy
    ...       
├── test
    ├──024vJIboFg4.npy
    ├──02rnKVawh0E.npy
    ...                
```

To make this, run the following code. You need to change the `feature_file`, `pickle_file` and `out_dir` on your own.

```python
python utils/make_npyfiles.py --feature_file FEATURE_FILE --pickle_file PICKLE_FILE --out_dir OUT_DIR
```

The parameters are expalined below.
```[bash]
feature_file : 'features.npy' that you generated from the previous step.
pickle_file : The pickle file that contains time intervals with narration_id and video_ids. AVE.TEST_LIST in above example.
out_dir : output directory to save the rearranged features. We recommend to put the split name ('train', 'val' or 'test) at the end. (ex - /path/to/outputs/{train,val,test})
```
