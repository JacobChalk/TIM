import glob
import subprocess
import os
import multiprocessing
import time
from itertools import repeat
import argparse

parser = argparse.ArgumentParser(description=('Extract frames for the Perception Test dataset'))
parser.add_argument('video_dir', type=str, help='Path to .MP4 videos')
parser.add_argument('out_dir', type=str, help='Path to save extracted frames')

def ffmpeg_extraction(videofile, out_dir):
    basename = os.path.basename(videofile)[:-4]
    out_dir = os.path.join(out_dir, basename)
    os.makedirs(out_dir, exist_ok=True)

    command = f"ffmpeg -i {videofile} '{out_dir}/frame_%10d.jpg'"
    subprocess.call(command, shell=True)

    time.sleep(1)

if __name__ == '__main__':
    args = parser.parse_args()

    mp4files = glob.glob(os.path.join(args.video_dir, '*.mp4'))

    with multiprocessing.Pool(40) as p:
        p.map(ffmpeg_extraction, zip(mp4files, repeat(args.out_dir)))
