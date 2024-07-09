# Video feature extraction

This repo contains the video feature extraction code for Epic-KITCHENS, Perception Test and AVE dataset.

## Environment

We recommend you to use conda environment.

```[bash]
conda create -n tim_video python=3.10
conda activate tim_video
pip install -r requirements.txt
```

## EPIC-KITCHENS-100

### Download the EPIC-KITCHENS-100 data

A download script is provided for the videos [here](https://github.com/epic-kitchens/download-scripts-100). You could also download the frames and corresponding videos.

**Contact:** [uob-epic-kitchens@bristol.ac.uk](mailto:uob-epic-kitchens@bristol.ac.uk)

### Necessary files before extracting the EPIC-KITCHENS-100 features

- directory path that contains the individual rgb frames for EPIC-KITCHENS-100
- yaml file for model / data configuration, found in `configs/EPIC-KITCHENS/OMNIVORE_feature.yaml`
- pickle file that has start and end time stamp for frame-level features

### Run the EPIC-KITCHENS-100 feature extraction

First, correctly configure you python path with `export PYTHONPATH=/path/to/TIM/feature_extractors/omnivore/omnivore:$PYTHONPATH`, then run the following command:

```[bash]
python tools/run_net.py \
  --cfg /path/to/config/EPIC-KITCHENS/OMNIVORE_feature.yaml \
  NUM_GPUS <num_gpus> \
  OUTPUT_DIR /path/to/output/dataset_split \
  EPICKITCHENS.VISUAL_DATA_DIR /path/to/epic_frames \
  EPICKITCHENS.TEST_LIST /path/to/EPIC_100_feature_interval_times \
  TEST.BATCH_SIZE <batch_size> \
  TEST.NUM_FEATURES <num_features>
```

The parameters are explained below:

```[bash]
NUM_GPUS : number of gpus you use for extraction.
OUTPUT_DIR : output directory to save the features for dataset split
EPICKITCHENS.VISUAL_DATA_DIR : directory path which contains the extracted frames for EPIC-KITCHENS-100
EPICKITCHENS.TEST_LIST : pickle file containing time intervals for each feature in the EPIC-KITCHENS-100 dataset split
TEST.BATCH_SIZE : batch size should be a multiple of NUM_GPUS
TEST.CHECKPOINT_FILE_PATH : pretrained checkpoint file path
TEST.NUM_FEATURES : Number of feature sets for each time segment. 1 for validation and test set (unaugmented) and 4 for train set (unaugmented+augmented sets).
```

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

To make this, run the following code. You need to change the `feature_list`, `metafile_list` and `out_dir` on your own.

```python
python utils/epic-kitchens/make_npyfiles.py
```

## Perception test

### Download the Perception Test data

<!-- You need to visit perception_test official website -->
Visit the official [perception_test website](https://github.com/google-deepmind/perception_test) to download the video files. To extract the frames from videos, use `utils/perception_test/extract_frames.py`.

### Necessary files before extracting Perception Test features

- directory path that contains the individual rgb frames for the Perception Test dataset
- yaml file for model / data configuration `configs/PERCEPTION_TEST/OMNIVORE_feature.yaml`
- pickle file that has start and end time stamp for frame-level features

### Run the Perception Test feature extraction

First, correctly configure you python path with `export PYTHONPATH=/path/to/TIM/feature_extractors/omnivore/omnivore:$PYTHONPATH`, then run the following command:

```[bash]
python tools/run_net.py \
--cfg /path/to/configs/PERCEPTION_TEST/OMNIVORE_feature.yaml \
NUM_GPUS <num_gpus> \
OUTPUT_DIR /path/to/output/dataset_split \
PERCEPTION.VISUAL_DATA_DIR /path/to/perception_test_frames \
PERCEPTION.TEST_LIST /path/to/PERCEPTION_feature_interval_times \
TEST.BATCH_SIZE <batch_size> \
TEST.NUM_FEATURES <num_features>
```

The parameters are explained below:

```[bash]
NUM_GPUS : number of gpus you use for extraction. For audio, 1 is enough since it doesn't take a lot of time.
OUTPUT_DIR : output directory to save the features for dataset split
PERCEPTION.VISUAL_DATA_DIR : directory path which contains the extracted frames for Perception Test
PERCEPTION.TEST_LIST : pickle file containing time intervals for each feature in the Perception Test dataset split
TEST.BATCH_SIZE : batch size should be a multiple of NUM_GPUS
TEST.CHECKPOINT_FILE_PATH : pretrained checkpoint file path
TEST.NUM_FEATURES : Number of feature sets for each time segment. 1 for validation and test set (unaugmented) and 4 for train set (unaugmented+augmented sets).
```

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

To make this, run the following code. You need to change the `feature_list`, `metafile_list` and `out_dir` on your own.

```python
python utils/perception_test/make_npyfiles.py
```

## AVE

### Download the AVE data

<!-- You need to visit AVE official website -->
Visit the official [AVE website](https://github.com/YapengTian/AVE-ECCV18) to download the video files.
To extract the frames, please refer to [utils/ave/extract_frames.py](utils/ave/extract_frames.py)

### Necessary files before extracting AVE features

- directory path that contains the individual rgb frames for the AVE dataset
- yaml file for model / data configuration `configs/AVE/OMNIVORE_feature.yaml`
- pickle file that has start and end time stamp for frame-level features

### Run AVE feature extraction

First, correctly configure you python path with `export PYTHONPATH=/path/to/TIM/feature_extractors/omnivore/omnivore:$PYTHONPATH`, then run the following command:

```[bash]
python tools/run_net.py \
--cfg /path/to/configs/AVE/OMNIVORE_feature.yaml \
NUM_GPUS <num_gpus> \
OUTPUT_DIR /path/to/output/dataset_split \
AVE.VISUAL_DATA_DIR /path/to/AVE_frames \
AVE.TEST_LIST /path/to/AVE_feature_interval_times \
TEST.BATCH_SIZE <batch_size> \
TEST.NUM_FEATURES <num_features>
```

The parameters are explained below:

```[bash]
NUM_GPUS : number of gpus you use for extraction.
OUTPUT_DIR : output directory to save the features for dataset split
AVE.VISUAL_DATA_DIR : directory path which contains the extracted frames for AVE
AVE.TEST_LIST : pickle file containing time intervals for each feature in the AVE dataset split
TEST.BATCH_SIZE : batch size should be a multiple of NUM_GPUS
TEST.CHECKPOINT_FILE_PATH : pretrained checkpoint file path
TEST.NUM_FEATURES : Number of feature sets for each time segment. 1 for validation and test set (unaugmented) and 4 for train set (unaugmented+augmented sets).
```

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

To make this, run the following code. You need to change the `feature_list`, `metafile_list` and `out_dir` on your own.

```[python]
python utils/ave/make_npyfiles.py
```
