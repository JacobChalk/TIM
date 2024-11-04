# This file is used to extract frames and compress to tar files from perception_test videos.
# Jaesung Huh, 2024

import glob
import subprocess
import os
import multiprocessing
import time
import tarfile

# Change these for your own paths
video_dir = '/scratch/shared/beegfs/jaesung/dataset/perceptiontest/data/videos'  # Directory that contains videofiles
frame_dir = '/scratch/shared/beegfs/jaesung/dataset/perceptiontest/data/frames'  # Output directory for frames
tar_dir = '/scratch/shared/beegfs/jaesung/dataset/perceptiontest/data/tarfiles'  # Output directory for tar files 


def ffmpeg_extraction(videofile):
    # Extract frames from the video file
    basename = os.path.basename(videofile)[:-4]
    outdir = os.path.join(frame_dir, basename)
    os.makedirs(outdir, exist_ok=True)

    command = f"ffmpeg -i {videofile} '{outdir}/frame_%10d.jpg'"
    subprocess.call(command, shell=True)

    time.sleep(1)

    # Make tarfile, you don't need to do this if you are using the frames directly
    imagefiles = glob.glob(os.path.join(outdir, '*.jpg'))
    basename = os.path.basename(outdir)
    tar_outfile = os.path.join(tar_dir, f'{basename}.tar')
    os.makedirs(os.path.dirname(tar_outfile), exist_ok=True)
    with tarfile.open(tar_outfile, 'w') as f:
        count = 0
        for imagefile in imagefiles:
            count += 1
            f.add(imagefile, arcname=f'{basename}/{os.path.basename(imagefile)}')


if __name__ == '__main__':
    mp4files = glob.glob(os.path.join(video_dir, '*.mp4'))

    with multiprocessing.Pool(40) as p:
        p.map(ffmpeg_extraction, mp4files)
