# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

''' Creates tfrecord file for inference '''

import argparse
from glob import glob
import os
import numpy as np
import tensorflow as tf
import json

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--folder', type=str, required=True, help = 'Folder with subfolders of json files')
    parse.add_argument('--fps', type=int, required=True, help='FPS of the video')
    args = parse.parse_args()
    return args

def get_json_data(video):
    '''
    Organizes human pose keypoints data to create tfrecord file.

    Input:
        folder (str): Path to folder with 1 json file for each frame of the video.
                      Each json file contains 137 keypoints for a person in the video.

    Output:
         skel_data (np.array): Array of shape (#frames, 1, 137, 2) with coordinates for each keypoint.
         conf_data (np.array): Array of shape (#frames, 1, 137) with human pose confidence estimation for each keypoint.
    '''

    # list all json files in video folder
    json_list = glob(video + '*')
    json_list.sort(key=os.path.getmtime)
    total_frames = len(json_list)

    # list number of keypoints for body part
    keypoint_types = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    body_points = {'pose_keypoints_2d': 25,
                  'face_keypoints_2d': 70,
                  'hand_left_keypoints_2d': 21,
                  'hand_right_keypoints_2d': 21}
    total_keypoints = 137

    # create empty numpy array
    skel_data = np.empty(shape=(total_frames, 1, total_keypoints, 2), dtype=np.float32)
    conf_data = np.empty(shape=(total_frames, 1, total_keypoints), dtype=np.float32)

    # for each frame (json file)
    frame = 0
    for file_name in json_list:
        # get json data
        file = open(file_name)
        data = json.load(file)
        skel = data['people'][0]

        # get all skeleton values
        count = 0
        for body_part in keypoint_types:
            num_points = body_points[body_part]
            for point in range(num_points):
                coord = point*3
                skel_data[frame, 0, count, 0] = skel[body_part][coord] # first coordinate
                skel_data[frame, 0, count, 1] = skel[body_part][coord+1] # second coordinate
                conf_data[frame, 0, count] = skel[body_part][coord+2] # confidence
                count = count + 1
        frame = frame + 1
    
    return skel_data, conf_data


if __name__ == '__main__':
    # get list of videos
    args = get_args()
    videos_list = glob(args.folder + '*/', recursive=True)

    # extract skeleton for each video
    for video in videos_list:
        # get data from json
        skel_data, conf_data = get_json_data(video)

        # create tfrecord for inference
        file_name = video[:-1] + '.tfrecord'
        print(file_name)
        with tf.io.TFRecordWriter(file_name) as writer:
            for video_id in range(1):
                
                data = tf.io.serialize_tensor(tf.convert_to_tensor(skel_data)).numpy()
                confidence = tf.io.serialize_tensor(tf.convert_to_tensor(conf_data)).numpy()

                features = {
                    'fps':
                        tf.train.Feature(int64_list=tf.train.Int64List(value=[args.fps])),
                    'pose_data':
                        tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
                    'pose_confidence':
                        tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence]))
                }

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
