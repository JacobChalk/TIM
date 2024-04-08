# TIM: A Time Interval Machine for Audio-Visual Action Recognition - Feature Extractors

## Requirements

The requirements vary based on the feature extractor used. Please check each folder individually.

## Replicating our EPIC-KITCHENS features

Once you have extracted both Omnivore and VideoMAE features, you will need to run the `merge_features.py` script to concatenate both along the channel dimension to replicate our features. Use the following command to combine the features:

```[bash]
python merge_features.py /path/to/omnivore_features /path/to/videomae_features --output_path /path/to/save/merged_features
```

## Reference

The code in the following subfolders are modified versions of the [Omnivore](https://github.com/beasteers/ego_actrecog_analysis), [Auditory SlowFast](https://github.com/ekazakos/auditory-slow-fast) and [VideoMAE](https://github.com/MCG-NJU/VideoMAE)/[InternVideo](https://github.com/OpenGVLab/InternVideo) repositories.
We also recommend you to read the papers below for details of each model:

```[bibtex]
@ARTICLE{Kazakos2021SlowFastAuditory,
   title={Slow-Fast Auditory Streams For Audio Recognition},
   author={Kazakos, Evangelos and Nagrani, Arsha and Zisserman, Andrew and Damen, Dima},
           journal   = {CoRR},
           volume    = {abs/2103.03516},
           year      = {2021},
           ee        = {https://arxiv.org/abs/2103.03516},
}
@inproceedings{girdhar2022omnivore,
  title={{Omnivore: A Single Model for Many Visual Modalities}},
  author={Girdhar, Rohit and Singh, Mannat and Ravi, Nikhila and van der Maaten, Laurens and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2022}
}
@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
@article{wang2022internvideo,
  title={InternVideo: General Video Foundation Models via Generative and Discriminative Learning},
  author={Wang, Yi and Li, Kunchang and Li, Yizhuo and He, Yinan and Huang, Bingkun and Zhao, Zhiyu and Zhang, Hongjie and Xu, Jilan and Liu, Yi and Wang, Zun and Xing, Sen and Chen, Guo and Pan, Junting and Yu, Jiashuo and Wang, Yali and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2212.03191},
  year={2022}
}
```
