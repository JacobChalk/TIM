import pandas as pd
import torch
import json
import subprocess
import numpy as np
import argparse

from joblib import Parallel, delayed
from tqdm import tqdm

from nms import batched_nms

parser = argparse.ArgumentParser(
    description="Evaluate EPIC-KITCHENS-100 validation proposals for detection from two stream model",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "path_to_verb_preds",
    help = "Path to the verb detection results from TIM"
)
parser.add_argument(
    "path_to_noun_preds",
    help = "Path to the noun detection results from TIM"
)
parser.add_argument(
    "path_to_gt",
    help = "Path to the ground truth files"
)
parser.add_argument(
    "--score_threshold",
    type=float,
    default=0.03,
    help = "Preprocessing score threshold"
)
parser.add_argument(
    "--verb_alpha",
    type=float,
    default=0.65,
    help = "Alpha used to balance between verb and noun outputs"
)
parser.add_argument(
    "--top_k",
    type=int,
    default=1,
    help = "Top k verb/noun predictions to compare to"
)
parser.add_argument(
    "--sigma",
    type=float,
    default=0.25,
    help = "Sigma for soft NMS"
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
            filter='action'
        ):
    nms_results = []
    segs = [d['segment'] for d in results_in_vid]
    scores = [d['score'] for d in results_in_vid]
    if filter == 'action':
        labels = [(d['verb'] * 300) + d['noun'] for d in results_in_vid]
    else:
        labels = [d[filter] for d in results_in_vid]

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
        score = float(scores[p].item())

        action = int(labels[p].item())
        if filter == 'action':
            verb = int(action // 300)
            noun = int(action % 300)
        else:
            verb = action
            noun = action

        action = f"{verb},{noun}"

        entry = {
                "verb": verb,
                "noun": noun,
                "action": action,
                "score": score,
                "segment": [round(start, 3), round(stop, 3)]
            }
        nms_results.append(entry)

    return nms_results, vid

def main(args):
    print("Loading Files")
    verb_outs = torch.load(args.path_to_verb_preds, map_location='cpu')
    noun_outs = torch.load(args.path_to_noun_preds, map_location='cpu')

    outs = {
            "video_ids": verb_outs["video_ids"],
            "verb": verb_outs['action'],
            "noun": noun_outs['action'],
            "verb_proposals": verb_outs['v_proposals'],
            "noun_proposals": noun_outs['v_proposals']
        }

    # print(f"Getting Scores and Predictions from {outs['video_ids'].shape[0]} proposals.")
    results = {}
    init_count = 0
    top_k = args.top_k
    for i in tqdm(range(outs["verb"].shape[0])):
        vid = str(outs["video_ids"][i])
        top_k_verb_inds = np.argpartition(outs['verb'][i], -top_k)[-top_k:]
        verb_scores = outs['verb'][i][top_k_verb_inds]

        top_k_noun_inds = np.argpartition(outs['noun'][i], -top_k)[-top_k:]
        noun_scores = outs['noun'][i][top_k_noun_inds]

        for v, verb_score in enumerate(verb_scores):
            if verb_score > args.score_threshold:
                for n, noun_score in enumerate(noun_scores):
                    if noun_score > args.score_threshold:
                        score = (verb_score**args.verb_alpha)*(noun_score**(1.0 - args.verb_alpha))
                        if score > args.score_threshold:
                            weight = verb_score / (verb_score + noun_score)
                            proposal = (weight * outs['verb_proposals'][i]) + ((1 - weight) * (outs['noun_proposals'][i]))
                            proposal = np.round(proposal, 3)
                            if (proposal[1] -  proposal[0] > 0.0):
                                    verb = top_k_verb_inds[v]
                                    noun = top_k_noun_inds[n]
                                    entry = {
                                        'verb': verb,
                                        'noun': noun,
                                        'action': f"{verb},{noun}",
                                        'score': score,
                                        'segment': [proposal[0], proposal[1]]
                                    }
                                    if vid in results:
                                        results[vid].append(entry)
                                    else:
                                        results[vid] = [entry]
                                    init_count += 1

    print((f'Creating Submission from {init_count} predictions.'))

    results = {k: v for k, v in sorted(results.items(), key=lambda item: len(item[1]))}

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(filter_nms)(
            results_in_vid=v,
            vid=k,
            iou_threshold=0.1,
            min_score=0.001,
            sigma=args.sigma,
            method=2,
            nms='soft'
        ) for k, v in tqdm(results.items()))

    results = {t[1]: t[0] for t in results}

    print("Total Entries:", sum([len(v) for k, v in results.items()]))

    results = {k: sorted(v, key=lambda x: x['score'], reverse=True) for k, v in results.items()}

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
    subprocess.run(["python", "evaluate_detection_json_ek100.py", f"tim.json", args.path_to_gt])
