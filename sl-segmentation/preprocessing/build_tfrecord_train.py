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

''' Creates tfrecord file for training '''

import argparse
from glob import glob
import os
import numpy as np
import tensorflow as tf
import json
import pympi

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--skel', type=str, required=True, help = 'Folder with subfolders of json files')
    parse.add_argument('--type_skel', type=str, required=True, help = 'Type of skeleton data ("MEDIAPI" or "DGS")')
    parse.add_argument('--annot', type=str, required=True, help = 'Folder with annotation files')
    parse.add_argument('--type_annot', type=str, required=True, help = 'TYpe of annotation files ("eaf" or "vtt")')
    parse.add_argument('--fps', type=int, required=True, help='FPS of the video')
    parse.add_argument('--output', type=str, required=True, help='Name of the tfrecord file', default='dataset')
    args = parse.parse_args()
    return args

def get_json_data(video, json_type):
    '''
    Organizes human pose keypoints data in numpy arrays.

    Input:
        video (str): Path to folder with json files (MEDIAPI-SKEL) or path to json file (DGS) with body, face, and hand keypoints.
        json_type (str): Identify how the json data is organized:
                            - 'MEDIAPI': One json file for each frame of the video (MEDIAPI, OpenPose default).
                            - 'DGS': One json file per video with keypoints for all frames (Public DGS Corpus format).

    Output:
        skel_data (list of np.arrays): List of arrays of shape (#frames, 1, 137, 2) with coordinates for each keypoint.
        conf_data (list of np.arrays): List of arrays of shape (#frames, 1, 137) with human pose confidence estimation for each keypoint.

        json_type == 'DGS' generates a list of max size 2. json_type == 'mediapi' generates a list of size 1.
    '''

    if json_type not in ('MEDIAPI','DGS'):
        print("'json_type' argument must be in ('MEDIAPI','DGS')")
        return 0
    
    # list number of keypoints for body part
    keypoint_types = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    body_points = {'pose_keypoints_2d': 25,
                   'face_keypoints_2d': 70,
                   'hand_left_keypoints_2d': 21,
                   'hand_right_keypoints_2d': 21}
    total_keypoints = 137

    ## one json file for each video frame
    if json_type == 'MEDIAPI':
        # list all json files in video folder
        json_list = glob(video + '*')
        json_list.sort(key=os.path.getmtime)
        total_frames = len(json_list)

        # create output list
        skel_list = []
        conf_list = []
        info = []

        # create empty numpy array
        skel_data = np.empty(shape=(total_frames, 1, total_keypoints, 2), dtype=np.float32)
        conf_data = np.empty(shape=(total_frames, 1, total_keypoints), dtype=np.float32)

        # for each frame (json file)
        frame = 0
        for file_name in json_list:
            # get json data
            file = open(file_name)
            data = json.load(file)
            skel = data['people'][0] # only first person

            # get all skeleton values
            count = 0
            for body_part in keypoint_types:
                num_points = body_points[body_part]
                for point in range(num_points):
                    coord = point*3
                    if len(skel[body_part]) > 0:
                        skel_data[frame, 0, count, 0] = skel[body_part][coord] # first coordinate
                        skel_data[frame, 0, count, 1] = skel[body_part][coord+1] # second coordinate
                        conf_data[frame, 0, count] = skel[body_part][coord+2] # confidence
                    # if data is missing, fill with zeros
                    else:
                        skel_data[frame_num, 0, count, 0] = 0 # first coordinate
                        skel_data[frame_num, 0, count, 1] = 0 # second coordinate
                        conf_data[frame_num, 0, count] = 0 # confidence
                    count = count + 1
            frame = frame + 1

        # add to output list
        info.append(('cam', total_frames))
        skel_list.append(skel_data)
        conf_list.append(conf_data)

    ## one json file for all the frames in the video (Public DGS Corpus)
    if json_type == 'DGS':
        # read json file
        file = open(video)
        data = json.load(file)
        num_cameras = len(data)

        # create output lists
        skel_list = []
        conf_list = []
        info = []

        # for each camera
        for cam_number in range(num_cameras):
            # get data from cameras A and B
            if data[cam_number]['camera'] in ('a1','b1'):
                # get list of frames
                frames_data = data[cam_number]['frames']
                frames_list = frames_data.keys()
                total_frames = int(list(frames_list)[-1]) + 1

                # create empty numpy array
                skel_data = np.zeros(shape=(total_frames, 1, total_keypoints, 2), dtype=np.float32)
                conf_data = np.zeros(shape=(total_frames, 1, total_keypoints), dtype=np.float32)

                # get all skeleton values
                for frame in frames_list:
                    skel = frames_data[frame]['people'][0] # only first person
                    count = 0
                    frame_num = int(frame)
                    for body_part in keypoint_types:
                        num_points = body_points[body_part]
                        for point in range(num_points):
                            coord = point*3
                            if len(skel[body_part]) > 0:
                                skel_data[frame_num, 0, count, 0] = skel[body_part][coord] # first coordinate
                                skel_data[frame_num, 0, count, 1] = skel[body_part][coord+1] # second coordinate
                                conf_data[frame_num, 0, count] = skel[body_part][coord+2] # confidence
                            # if data is missing, fill with zeros
                            else:
                                skel_data[frame_num, 0, count, 0] = 0 # first coordinate
                                skel_data[frame_num, 0, count, 1] = 0 # second coordinate
                                conf_data[frame_num, 0, count] = 0 # confidence
                            count = count + 1

                # add to output list
                info.append((data[cam_number]['camera'], total_frames))
                skel_list.append(skel_data)
                conf_list.append(conf_data)
    
    return skel_list, conf_list, info


def get_annotations(file, fps, info, annot_type):
    '''
    Tranform sign language video annotations in binary variable (signing/not signing)

    Input:
        video (str): Path to file with annotations.
        annot_type (str): Format of the annotation file:
                            - 'eaf': Annotation file from ELAN software from the Public DGS Corpus
                            - 'vtt': Subtitle file format from the MEDIAPI-SKEL dataset

    Output:
        labels (list of np.arrays): List of arrays of shape (#frames) with binary variable indicating if the person is signing at each frame.
        error_list (list): List of indexes where no annotation was found.

    annot_type == 'eaf' generates a list of max size 2. annot_type == 'vtt' generates a list of size 1.
    '''

    if annot_type not in ('eaf','vtt'):
        print("'annot_type' argument must be in ('eaf','vtt')")
        return 0
    
    # ELAN files
    if annot_type == 'eaf':
        # cameras in json file
        tiers_list = []
        for i in range(len(info)):
            if info[i][0] == 'a1':
                tiers_list.append('Deutsche_Übersetzung_A') # tiers in DGS
            elif info[i][0] == 'b1':
                tiers_list.append('Deutsche_Übersetzung_B') # tiers in DGS
        
        eaf = pympi.Elan.Eaf(file)
        file_tiers = list(eaf.tiers.keys()) # tiers on the file

        labels = []
        error_list = []
        index = 0
        for tier_name in tiers_list:
            # if tier exists in the file
            if tier_name in file_tiers:
                start = []
                end = []
                # extract start and end of annotations
                tier = eaf.tiers[tier_name][0]
                for id in tier.keys():
                    start.append(round(eaf.timeslots[tier[id][0]]/1000) * fps)
                    end.append((round(eaf.timeslots[tier[id][1]]/1000) * fps) - round(0.06*fps)) # at least 0.06 seconds between annotations

                # build array
                annotation = np.zeros(shape=(info[index][1]), dtype='byte')
                for j in range(len(start)):
                    pointer = start[j]
                    while (pointer <= end[j]) and (pointer < annotation.shape[0]):
                        annotation[pointer] = 1
                        pointer = pointer + 1
                
                labels.append(annotation)
                index = index + 1
                
            # if tier doesn't exist in file, notify 
            else:
                labels.append(None)
                without_annot = tiers_list.index(tier_name)
                error_list.append(without_annot)
            	

        # vtt files (MEDIAPI) - NOT FINISHED
        if annot_type == 'vtt':
            labels = []

        return labels, error_list


if __name__ == '__main__':
    args = get_args()

    # list all videos
    if args.type_skel == 'MEDIAPI':
        videos_list = glob(args.skel + '*/')
    elif args.type_skel == 'DGS':
        videos_list = glob(args.skel + '*.json')

    # create tfrecord file for training
    file_name = args.output + '.tfrecord'
    count = 0
    print('Processing...')
    with tf.io.TFRecordWriter(file_name) as writer:
        for video in videos_list:
            # get skeleton data
            skel_data, conf_data, info = get_json_data(video, json_type=args.type_skel)
    
            # get annotation data
            if args.type_skel == 'MEDIAPI':
                annot_file = '00001.fr.vtt' # fix later
            elif args.type_skel == 'DGS':
                annot_file = args.annot + video[len(args.skel):-14] + '.eaf' # Public DGS Corpus deafult name
            
            annot_data, error_list = get_annotations(file=annot_file, fps=args.fps, info=info, annot_type=args.type_annot)

            # organize data for each camera
            for i in range(len(skel_data)):
            	if i not in error_list:
                    data = tf.io.serialize_tensor(tf.convert_to_tensor(skel_data[i])).numpy()
                    confidence = tf.io.serialize_tensor(tf.convert_to_tensor(conf_data[i])).numpy()
                    is_signing = annot_data[i].tobytes()
    
                    features = {
                        'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.fps])),
                        'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
                        'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[confidence])),
                        'is_signing': tf.train.Feature(bytes_list=tf.train.BytesList(value=[is_signing]))
                    }

                    # write to file
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())
                    
            count = count + 1
            print(f'{round(100*count/len(videos_list))}% processed.')
    
    print(f'tfrecord created: {file_name}')
