import numpy as np
import tqdm
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description=('Create .npy files for the EPIC-KITCHENS dataset'))

###### Things you need to modify ######
parser.add_argument('--feature_list', type=str, help='Path to extracted features')
parser.add_argument('--metafile_list', type=str, help='Path to feature time intervals')
parser.add_argument('--out_dir', type=str, help='Path to save features to')
#######################################

def main(args):
    result_dict = {}
    print("Parsing the features")
    feature = np.load(args.feature_list,)
    metadata = pd.read_pickle(args.metafile_list)
    for i in tqdm.tqdm(range(len(metadata))):
        annotation_id = metadata.iloc[i]['narration_id']
        vid_id = metadata.iloc[i]['video_id']
        if vid_id not in result_dict:
            result_dict[vid_id] = {}
        annotation_index = int(annotation_id.split('_')[-1])
        new_annotation_id = '{}_{:06d}'.format(vid_id, annotation_index)
        if new_annotation_id not in result_dict[vid_id]:
            result_dict[vid_id][new_annotation_id] = []
        
        result_dict[vid_id][new_annotation_id].append(feature[i])

    # SAVE npyfiles
    print("Saving as npy files")

    for vid_id, v_features in tqdm.tqdm(result_dict.items()):
        args.feature_list = []
        annotation_ids = sorted(list(v_features.keys()))
        for annotation_id in annotation_ids:
            feature = np.stack(v_features[annotation_id], axis=0)
            args.feature_list.append(feature)
        out_file = os.path.join(args.out_dir, f'{vid_id}.npy')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        vid_feature = np.stack(args.feature_list, axis=0)
        if len(vid_feature.shape) == 4:
            vid_feature = np.squeeze(vid_feature, axis=1)
        np.save(out_file, vid_feature)

if __name__ == "__main__":
    main(parser.parse_args())