# TIM: A Time Interval Machine for Audio-Visual Action Recognition - Feature Extractors

## Requirements

The requirements vary based on the feature extractor used. Please check each folder individually.

## Replicating our EPIC-KITCHENS features

Once you have extracted both Omnivore and VideoMAE features, you will need to run the `merge_features.py` script to concatenate both along the channel dimension to replicate our features. Use the following command to combine the features:

```[bash]
python merge_features.py /path/to/omnivore_features /path/to/videomae_features --output_path /path/to/save/merged_features
```
