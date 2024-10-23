import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description=('Merge Omnivore and VideoMAE features'))
parser.add_argument('omnivore_feature_path', type=str)
parser.add_argument('videomae_feature_path', type=str)
parser.add_argument('--output_path', type=str, default='./videovore')



def main(args):
    """Merge Omnivore and VideoMAE feature into [N, num_aug, 2048] features. 
        Assumes file structure of:
            omnivore_feature_path
            |
            |_ train
            |   |
            |   |_ train_video_1.npy
            |   |
            |   |_ train_video_2.npy
            |
            |_ val
               |
               |_ validation_video_1.npy
               |
               |_ validation_video_2.npy

        and:

            videomae_feature_path
            |
            |_ train
            |   |
            |   |_ train_video_1.npy
            |   |
            |   |_ train_video_2.npy
            |
            |_ val
               |
               |_ validation_video_1.npy
               |
               |_ validation_video_2.npy

        The script will output concatenate feature in the same structure as above

    Args:
        args (Namespace): Contains arguments for both sets of feature paths as well as output dir
    """
    omnivore_splits = os.listdir(args.omnivore_feature_path)
    videomae_splits = os.listdir(args.videomae_feature_path)

    matched_splits = [folder for folder in omnivore_splits if folder in videomae_splits]

    assert len(matched_splits) == 0, "No matching splits found. Ensure the features are stored as <backbone>/{train,val,test}/<video>.npy"

    for split in matched_splits:
        omnivore_features = os.listdir(f"{args.omnivore_feature_path}/{split}")
        videomae_features = os.listdir(f"{args.videomae_feature_path}/{split}")

        assert len(omnivore_features) == len(videomae_features), f"Mismatch in number of features for split: {split}. Omnivore: {len(omnivore_features)} != VideoMAE: {len(videomae_features)}"

        if not os.path.exists(f"{args.output_path}/{split}"):
            os.makedirs(f"{args.output_path}/{split}", exist_ok=True)

        for feature in omnivore_features:
            if ".npy" in feature:
                omnivore_feature_path = f"{args.omnivore_feature_path}/{split}/{feature}"
                videomae_feature_path = f"{args.videomae_feature_path}/{split}/{feature}"
                omnivore_feature = np.load(omnivore_feature_path)
                videomae_feature = np.load(videomae_feature_path)

                if omnivore_feature.ndim == 2:
                    omnivore_feature = np.expand_dims(omnivore_feature, axis=1)

                if videomae_feature.ndim == 2:
                    videomae_feature = np.expand_dims(videomae_feature, axis=1)
                
                assert omnivore_feature.shape[1] == videomae_feature.shape[1], f"Mismatch in number of feature sets for feature: {feature}. Omnivore: {omnivore_feature.shape[1]} != VideoMAE: {videomae_feature.shape[1]}"
                assert omnivore_feature.shape[-1] == 1024, f"Omnivore feature ({feature}) does not match expected shape [N, num_aug, 1024]. Got {omnivore_feature.shape}"
                assert videomae_feature.shape[-1] == 1024, f"VideoMAE feature ({feature}) does not match expected shape [N, num_aug, 1024]. Got {videomae_feature.shape}"

                videovore_feature = np.concatenate([omnivore_feature, videomae_feature], axis=-1)
                np.save(f"{args.output_path}/{split}/{feature}", videovore_feature)
            else:
                print(f"Skipping: {feature}")

if __name__ == '__main__':
    main(parser.parse_args())
