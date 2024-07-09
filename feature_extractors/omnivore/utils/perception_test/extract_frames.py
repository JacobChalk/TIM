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

def ffmpeg_extraction(videofile):
    basename = os.path.basename(videofile)[:-4]

    # Change this to store the frames
    outdir = f'{args.out_dir}/{basename}'
    os.makedirs(outdir, exist_ok=True)

    command = f"ffmpeg -i {videofile} '{outdir}/frame_%10d.jpg'"
    subprocess.call(command, shell=True)

    time.sleep(1)

if __name__ == '__main__':
    args = parser.parse_args()

    # Change the path to your own dataset path
    mp4files = glob.glob(f'{args.video_dir}*.mp4')

    with multiprocessing.Pool(40) as p:
        p.map(ffmpeg_extraction, mp4files)
