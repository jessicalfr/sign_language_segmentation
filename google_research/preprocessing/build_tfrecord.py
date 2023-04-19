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

"""Create tfrecord for training."""

import numpy as np
import tensorflow as tf
import json

def array_dgs_corpus(file):
    """
    Creates a np.array given a data input from DGS Corpus data.
    """

    # get json data
    data = json.load(file)

    # list number of keypoints for body part
    keypoint_types = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    body_points = {'pose_keypoints_2d': 25,
                  'face_keypoints_2d': 70,
                  'hand_left_keypoints_2d': 21,
                  'hand_right_keypoints_2d': 21}
    total_keypoints = 137

    # get frames data from camera A
    cam_a = data[0]
    frames = cam_a['frames']
    frames_list = list(frames.keys())
    total_frames = len(frames_list)

    # create empty numpy array
    skel_data = np.empty(shape=(total_frames, 1, total_keypoints, 2))
    conf_data = np.empty(shape=(total_frames, 1, total_keypoints))

    # create np.array
    for frame in frames_list:
        skel = frames[frame]['people'][0] # only the first person
        frame_num = int(frame)
        count = 0
        for body_part in keypoint_types:
            num_points = body_points[body_part]
            for point in range(num_points):
                coord = point*3
                skel_data[frame_num, 0, count, 0] = skel[body_part][coord] # first coordinate
                skel_data[frame_num, 0, count, 1] = skel[body_part][coord+1] # second coordinate
                conf_data[frame_num, 0, count] = skel[body_part][coord+2] # confidence
                count = count + 1

    return skel_data, conf_data


if __name__ == '__main__':

    # read json with openpose skeleton
    file = open('1413451-11105600-11163240_openpose.json')
    fps = 50

    # get data from json
    skel_data, conf_data = array_dgs_corpus(file)

    # create tfrecord
    with tf.io.TFRecordWriter('example.tfrecord') as writer:
        for video_id in range(1):
            
            data = tf.io.serialize_tensor(tf.convert_to_tensor(skel_data)).numpy()
            confidence = tf.io.serialize_tensor(tf.convert_to_tensor(conf_data)).numpy()

            features = {
                'fps':
                    tf.train.Feature(int64_list=tf.train.Int64List(value=[fps])),
                'pose_data':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
                'pose_confidence':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence]))
            }

            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
