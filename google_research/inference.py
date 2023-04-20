import functools
import os
from typing import Any
from typing import Dict
from typing import Tuple

from pose_format.pose import Pose
from pose_format.pose_header import PoseHeader
from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.tensorflow.pose_body import TF_POSE_RECORD_DESCRIPTION
from pose_format.utils.reader import BufferReader

import tensorflow as tf
from tensorflow import keras

from args import FLAGS


@functools.lru_cache(maxsize=1)
def get_openpose_header():
  """Get pose header with OpenPose components description."""
  dir_path = os.path.dirname(os.path.realpath(__file__))
  header_path = os.path.join(dir_path, "assets/openpose.poseheader")
  f = open(header_path, "rb")
  reader = BufferReader(f.read())
  header = PoseHeader.read(reader)
  return header


def differentiate_frames(src):
  """Subtract every two consecutive frames."""
  # Shift data to pre/post frames
  pre_src = src[:-1]
  post_src = src[1:]

  # Differentiate src points
  src = pre_src - post_src

  return src


def distance(src):
  """Calculate the Euclidean distance from x:y coordinates."""
  square = src.square()
  sum_squares = square.sum(dim=-1).fix_nan()
  sqrt = sum_squares.sqrt().zero_filled()
  return sqrt


def optical_flow(src, fps):
  """Calculate the optical flow norm between frames, normalized by fps."""

  # Remove "people" dimension
  src = src.squeeze(1)

  # Differentiate Frames
  src = differentiate_frames(src)

  # Calculate distance
  src = distance(src)

  # Normalize distance by fps
  src = src * fps

  return src
  
minimum_fps = tf.constant(1, dtype=tf.float32)

def load_datum(tfrecord_dict):
  """Convert tfrecord dictionary to tensors."""
  pose_body = TensorflowPoseBody.from_tfrecord(tfrecord_dict)
  pose = Pose(header=get_openpose_header(), body=pose_body)

  fps = pose.body.fps
  frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

  # Get only relevant input components
  pose = pose.get_components(FLAGS.input_components)

  return {
      "fps": pose.body.fps,
      "frames": frames,
      "pose_data_tensor": pose.body.data.tensor,
      "pose_data_mask": pose.body.data.mask,
      "pose_confidence": pose.body.confidence,
  }
  
def process_datum(datum,
                  augment=False):
  """Prepare every datum to be an input-output pair for training / eval.
  Supports data augmentation only including frames dropout.
  Frame dropout affects the FPS, which does change the optical flow.
  Args:
      datum (Dict[str, tf.Tensor]): a dictionary of tensors loaded from the
        tfrecord.
      augment (bool): should apply data augmentation on the datum?
  Returns:
     src tensors
  """
  #masked_tensor = MaskedTensor(
  #    tensor=datum["pose_data_tensor"], mask=datum["pose_data_mask"])
  pose_body = TensorflowPoseBody(
      fps=datum["fps"], data=datum["pose_data"], confidence=datum["pose_confidence"])
  pose = Pose(header=get_openpose_header(), body=pose_body)

  fps = pose.body.fps
  frames = datum["frames"]

  flow = optical_flow(pose.body.data, fps)

  return flow

# load model
model = keras.models.load_model('models/py/model.h5', compile=False)

# read tfrecord file
dataset = tf.data.TFRecordDataset(filenames=['example.tfrecord'])
features = TF_POSE_RECORD_DESCRIPTION # https://github.com/sign-language-processing/pose/blob/master/src/python/pose_format/tensorflow/pose_body.py
dataset = dataset.map(lambda serialized: tf.io.parse_single_example(serialized, features))
print(dataset)
flow = dataset.map(process_datum)
print(flow)

#pred = model.predict(flow)
