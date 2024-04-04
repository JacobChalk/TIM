# TIM: A Time Interval Machine for Audio-Visual Action Recognition

This repository provides the code used to implement the model proposed in the paper:

Jacob Chalk, Jaesung Huh, Evangelos Kazakos, Andrew Zisserman, Dima Damen, TIM: A Time Interval Machine for Audio-Visual Action Recognition, CVPR, 2024

[Project Webpage](https://jacobchalk.github.io/TIM-Project)

[ArXiv Paper]()

## Citing

When using this code, please reference:

```
PENDING
```

## Requirements

The requirements for TIM can be installed in a separate conda environment by running the following command in your terminal: `conda env create -f environment.yml`. You can then activate this with `conda activate TIM`.

**NOTE:** This environment only applies to the `recognition` and `detection` folders. Seperate requirements are listed for the backbones in the `feature_extractors` folder.

## Features

The features used for this project can be extracted by following the instructions in the `feature_extractors` folder.

## Pretrained models

You can find links to the relevant pre-trained models in the recognition and detection folders.

## Ground-Truth

We provide the necessary ground-truth files for all datasets here:

**NOTE:** These annotation files have been preprocessed to be compatible with the TIM codebase

## Training and Evaluating TIM

We provide instructions on how to train and evaluate TIM for both recognition and detection in the respective folders.

## License

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
