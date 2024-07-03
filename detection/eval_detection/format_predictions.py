import pandas as pd
import torch
import json
import subprocess
import numpy as np
import os
import argparse

from joblib import Parallel, delayed
from tqdm import tqdm

from nms import batched_nms

parser = argparse.ArgumentParser(
    description="Evaluate Perception Test/EPIC-Sounds validation proposals for detection",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "path_to_preds",
    help = "Path to the detection results from TIM"
)
parser.add_argument(
    "path_to_gt",
    help = "Path to the ground truth files"
)
parser.add_argument(
    "--score_threshold",
    type=float,
    default=0.01,
    help = "Preprocessing score threshold"
)
parser.add_argument(
    "--sigma",
    type=float,
    default=0.1,
    help = "Sigma for soft NMS"
)
parser.add_argument(
    "--is_audio",
    action='store_true',
    help="Flag to signify that the model being evaluated is for audio"
)
parser.add_argument(
    "--n_jobs",
    type=int,
    default=32,
    help="Number of parallel jobs to process detections"
)

def filter_nms(
            results_in_vid,
            vid,
            iou_threshold=0.1,
            min_score=0.001,
            sigma=0.4,
            method=2,
            nms='vanilla',
            multi_class=True,
            voting_thresh=0.75,
        ):
    nms_results = []
    segs = [d['segment'] for d in results_in_vid]
    scores = [d['score'] for d in results_in_vid]
    labels = [d['action'] for d in results_in_vid]

    segs = torch.FloatTensor(segs)
    scores = torch.FloatTensor(scores)
    labels = torch.LongTensor(labels)

    segs, scores, labels = batched_nms(
                        segs,
                        scores,
                        labels,
                        iou_threshold=iou_threshold,
                        min_score=min_score,
                        sigma=sigma,
                        method=method,
                        nms=nms,
                        multi_class=multi_class,
                        voting_thresh=voting_thresh
                    )

    for p in range(segs.shape[0]):
        start = float(segs[p][0])
        stop = float(segs[p][1])
        score = float(scores[p])

        action = int(labels[p])

        entry = {
                "action": action,
                "score": score,
                "segment": [round(start, 3), round(stop, 3)]
            }
        nms_results.append(entry)

    return nms_results, vid


def main(args):
    print("Loading Files")
    outs = torch.load(args.path_to_preds, map_location='cpu')

    print(f"Getting Scores and Predictions from {outs['video_ids'].shape[0]} proposals.")
    results = {v: [] for v in np.unique(outs['video_ids'])}
    init_count = 0
    multi_pred_size = 0
    multi_pred_count = 0
    
    for i in tqdm(range(outs['video_ids'].shape[0])):
        vid = str(outs["video_ids"][i])
        proposal = outs['a_proposals'][i] if args.is_audio else outs['v_proposals'][i]
        proposal = np.round(proposal, 3)
        if (proposal[1] -  proposal[0] > 0.0):
            scores = outs["audio"][i] if args.is_audio else outs['action'][i]
            valid_preds = np.where(scores > args.score_threshold)[0]
            if valid_preds.shape[0] > 0:
                multi_pred_size += valid_preds.shape[0]
                multi_pred_count += 1

                entries = [{
                    'action': pred,
                    'score':  scores[pred],
                    'segment': [proposal[0], proposal[1]]
                } for pred in valid_preds]
                results[vid].extend(entries)
                init_count += len(entries)

    print(f"Creating Submission from {init_count} predictions. Average Multi-Pred: {round(multi_pred_size / multi_pred_count, 2)}")

    results = {k: v for k, v in sorted(results.items(), key=lambda item: len(item[1]))}

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(filter_nms)(
            results_in_vid=v,
            vid=k,
            iou_threshold=0.1,
            min_score=0.001,
            sigma=args.sigma,
            method=2,
            nms='soft',
        ) for k, v in tqdm(results.items()))

    results = {t[1]: t[0] for t in results}

    print("Total Entries:", sum([len(v) for k, v in results.items()]))

    results = {k: sorted(v, key=lambda x: x['score'], reverse=True) for k, v in results.items()}

    output = {
            'video_id': [],
            'start': [],
            'stop': [],
            'action': [],
            'score': [],
        }

    for k, v in results.items():
        for i in range(len(v)):
            output['video_id'].append(k)
            output['start'].append(v[i]['segment'][0])
            output['stop'].append(v[i]['segment'][1])
            output['action'].append(v[i]['action'])
            output['score'].append(v[i]['score'])
    out_df = pd.DataFrame.from_dict(output)
    out_df = out_df.sort_values(['video_id', 'start'])
    out_df = out_df.reset_index(drop=True)
    print(out_df)


    submission = {
            "version": "0.2",
            "challenge": "action_detection",
            "sls_pt": 2,
            "sls_tl": 3,
            "sls_td": 4,
            "results": results
        }

    with open(f"tim.json", 'w') as f:
        json.dump(submission, f, indent=4, separators=(',', ': '))

    # Run the other script
    subprocess.run(["python", "evaluate_detection_json.py", f"tim.json", args.path_to_gt])

if __name__ == "__main__":
    main(parser.parse_args())
