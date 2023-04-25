import argparse
from glob import glob
import os

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--input_folder', '-i', type=str, required=True, help='Folder with videos to be processed')
    parse.add_argument('--output_folder', '-o', type=str, required=True, help='Folder where keypoints will be saved')
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # list all mp4 videos in folder
    videos_list = glob(args.input_folder + '*')

    # for each video, call openpose
    for video in videos_list:
        print(f'Processing video {video}...')
        video_name = video.split('/')
        new_path = args.output_folder + video_name[-1][:-4]
        os.system(f'/usr/local/bin/openpose.bin --video {video} --face --hand --write_json {new_path} --display 0 --render_pose 0 --model_folder /usr/local/openpose/models/')
        print(f'Keypoints saved in {new_path}')