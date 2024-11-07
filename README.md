# TIM: A Time Interval Machine for Audio-Visual Action Recognition

This repository provides the code used to implement the model proposed in the paper:

Jacob Chalk\*, Jaesung Huh\*, Evangelos Kazakos, Andrew Zisserman, Dima Damen, TIM: A Time Interval Machine for Audio-Visual Action Recognition, CVPR, 2024

(\*: Indicates equal contribution.)

[Project Webpage](https://jacobchalk.github.io/TIM-Project)

[ArXiv Paper](https://arxiv.org/abs/2404.05559)

## Citing

When using this code, please reference:

```[bibtex]
@InProceedings{Chalk_2024_CVPR,
    author    = {Chalk, Jacob and Huh, Jaesung and Kazakos, Evangelos and Zisserman, Andrew and Damen, Dima},
    title     = {{TIM}: {A} {T}ime {I}nterval {M}achine for {A}udio-{V}isual {A}ction {R}ecognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18153-18163}
}
```

## Requirements

The requirements for TIM can be installed in a separate conda environment by running the following command in your terminal: `conda env create -f environment.yml`. You can then activate this with `conda activate TIM`.

**NOTE:** This environment only applies to the `recognition` and `detection` folders. Seperate requirements are listed for the backbones in the `feature_extractors` folder.

## Features

The features used for this project can be extracted by following the instructions in the `feature_extractors` folder.

## Pre-trained models

You can find links to the relevant pre-trained models in the recognition, feature_extractors and detection folders.

## Ground-Truth

We provide the necessary ground-truth files for all datasets [here](https://www.dropbox.com/scl/fi/xs6muwf67a5h9ql30jart/annotations.zip?rlkey=iw6b4w9n4brcpvygoksmrvf4n&st=j6c1exut&dl=0).

The link contains a zip containing ground truth data for each dataset, consisting of:

- The training split ground truth
- The validation split ground truth
- The video metadata of the dataset
- The feature time intervals for training and valdiation splits

**NOTE:** These annotation files have been altered to be compatible with the TIM codebase.

## Training and Evaluating TIM

We provide instructions on how to train and evaluate TIM for both recognition and detection in the respective folders.

## License

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
