# This script makes video info pickle file.
# WARNING : This is the script that I used for AVE dataset. 
# You could use this for other datasets (e.g. Perception Test) with the same format.
# Jaesung Huh, 2024


from datetime import timedelta
import os
import csv
import pandas as pd
import cv2
import tqdm


## Change these paths to your own
annot_files = [
    '/scratch/shared/beegfs/jaesung/dataset/AVE_dataset/trainSet.txt',
    '/scratch/shared/beegfs/jaesung/dataset/AVE_dataset/valSet.txt',
    '/scratch/shared/beegfs/jaesung/dataset/AVE_dataset/testSet.txt'
]
video_dir = '/scratch/shared/beegfs/jaesung/dataset/AVE_dataset/video'  # Directory of video files
out_csvfile = '/scratch/shared/beegfs/jaesung/dataset/AVE_dataset/annotations/AVE_video_info.csv'
out_pklfile = '/scratch/shared/beegfs/jaesung/dataset/AVE_dataset/annotations/AVE_video_info.pkl'


def sec_to_timestamp(sec):
    undersecond = int((sec - int(sec)) * 100)
    a = timedelta(seconds=int(sec))
    return str(a) + '.' + '{:02d}'.format(undersecond)


def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = hours * 3600.00 + minutes * 60.0 + seconds
    return total_seconds


# This is only for AVE dataset. Change this for other datasets.
vid_ids = []
for annot_file in annot_files:
    with open(annot_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            cls_name, vid_id, qual, start, end = line.split('&')
            if vid_id not in vid_ids:
                vid_ids.append(vid_id)


# Write csv file
with open(out_csvfile, 'w') as f:
    writer = csv.writer(f)
    header = ['video_id', 'duration', 'fps']
    writer.writerow(header)
    for vid_id in tqdm.tqdm(vid_ids):
        videofile = os.path.join(video_dir, f'{vid_id}.mp4')
        cap = cv2.VideoCapture(videofile)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        video_length = 10.0  # All videos are 10 seconds in AVE dataset. Change this for other datasets.
        row = [vid_id, video_length, fps]
        writer.writerow(row)

# Write pickle file
df = pd.read_csv(out_csvfile)
df = df.set_index('video_id')
df.to_pickle(out_pklfile)