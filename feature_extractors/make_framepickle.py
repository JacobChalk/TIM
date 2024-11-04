# This file is used to make a pickle file for frame features.
# Jaesung Huh, 2024

import pandas as pd
import glob
import os
from datetime import timedelta
import time
import csv
import tqdm
import cv2

def timestamp_to_sec(timestamp):
    """Convert timestamp string to seconds."""
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(
        hours=x.tm_hour,
        minutes=x.tm_min,
        seconds=x.tm_sec
    ).total_seconds())
    return sec + float(timestamp.split('.')[-1][:2]) / 100


def sec_to_timestamp(sec):
    """Convert seconds to timestamp string."""
    undersecond = int((sec - int(sec)) * 100)
    time_delta = timedelta(seconds=int(sec))
    return f"{time_delta}.{undersecond:02d}"


# Change this for your own dataset
FRAME_DIR = '/scratch/shared/beegfs/jaesung/dataset/perceptiontest/data/frames'
OUT_CSV_FILE = '/scratch/shared/beegfs/jaesung/dataset/perceptiontest/annotations/frame_features/perception_frame_soundloc_val.csv'
OUT_PKL_FILE = '/scratch/shared/beegfs/jaesung/dataset/perceptiontest/annotations/frame_features/perception_frame_soundloc_val.pkl'
VIDEO_DIR = '/scratch/shared/beegfs/jaesung/dataset/perceptiontest/data/videos'
ID_CSV_FILE = '/scratch/shared/beegfs/jaesung/dataset/perceptiontest/data/localisation_challenge_valid_id_list.csv'
INTERVAL = 1.1
HOP_SIZE = 0.2

with open(ID_CSV_FILE, 'r') as f:
    lines = f.readlines()
    line = lines[0].strip()
    video_ids = line.split(',')


with open(OUT_CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'narration_id', 
        'video_id',
        'start_sec', 
        'stop_sec', 
        'narration_sec', 
        'start_frame', 
        'stop_frame'
    ])
    for video_id in tqdm.tqdm(video_ids):
        videofile = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        # Read the frame rate
        cam = cv2.VideoCapture(videofile)
        fps = cam.get(cv2.CAP_PROP_FPS)
        cam.release()
        # Check the length of the video
        framelist = glob.glob(os.path.join(FRAME_DIR, video_id, '*.jpg'))
        video_length = len(framelist) / fps
        start = 0
        index = 1
        while (start + INTERVAL) < video_length:
            start_timestamp = sec_to_timestamp(start)
            stop_timestamp = sec_to_timestamp(start + INTERVAL)
            start_frame = int(round(timestamp_to_sec(start_timestamp) * fps))
            stop_frame = int(round(timestamp_to_sec(stop_timestamp) * fps))
            narration_id = f'{video_id}_{index}'
            row = [
                narration_id,
                video_id,
                f'{start:.2f}',
                f'{start + INTERVAL:.2f}',
                f'{start + INTERVAL / 2:.2f}',
                start_frame,
                stop_frame
            ]
            writer.writerow(row)
            start += HOP_SIZE
            index += 1

df = pd.read_csv(OUT_CSV_FILE)
df.set_index('narration_id', inplace=True)
df.to_pickle(OUT_PKL_FILE)
