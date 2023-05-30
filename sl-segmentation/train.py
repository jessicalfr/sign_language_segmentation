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

"""Training script for sign language detection."""

import random

from absl import app
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import load_model

from args import FLAGS
from dataset import get_datasets
from model import build_model

import matplotlib.pyplot as plt
import numpy as np
import gc

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def set_seed():
  """Set seed for deterministic random number generation."""
  seed = FLAGS.seed if FLAGS.seed is not None else random.randint(0, 1000)
  tf.random.set_seed(seed)
  random.seed(seed)


def main(unused_argv):
  """Keras training loop with early-stopping and model checkpoint."""

  set_seed()

  tf.config.run_functions_eagerly(True)

  # set model name
  model_name = FLAGS.model_path.split('/')[-1].split('.')[0]

  # Initialize Dataset
  train, dev, test = get_datasets()

  # Initialize Model
  model = build_model()

  # Train
  es = EarlyStopping(
      monitor='val_accuracy',
      mode='max',
      verbose=1,
      patience=FLAGS.stop_patience)
  mc = ModelCheckpoint(
      FLAGS.model_path,
      monitor='val_accuracy',
      mode='max',
      verbose=1,
      save_best_only=True)
  hs = History()
  cl = ClearMemory()

  print('\nTraining:')
  with tf.device(FLAGS.device):
    model.fit(
        train,
        epochs=FLAGS.epochs,
        steps_per_epoch=FLAGS.steps_per_epoch,
        validation_data=dev,
        callbacks=[es, mc, hs, cl])

  # save history
  max_accuracy = hs.history['val_accuracy'].index(max(hs.history['val_accuracy']))

  plt.plot(hs.history['accuracy'])
  plt.plot(hs.history['val_accuracy'])
  plt.axvline(x=max_accuracy, linestyle='--', color='gray')
  plt.title('Model')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['train','val'], loc='lower right')
  plt.savefig(f'./results/plots/{model_name}.png')

  # save loss and accuracy results
  print('\nEvaluation:')
  print('\n - Train eval:')
  result_train = {'loss': hs.history['loss'][max_accuracy], 'accuracy': hs.history['accuracy'][max_accuracy]}
  print(result_train)

  print('\n - Dev eval:')
  result_dev = {'loss': hs.history['val_loss'][max_accuracy], 'accuracy': hs.history['val_accuracy'][max_accuracy]}
  print(result_dev)

  print('\n - Test eval:')
  best_model = load_model(FLAGS.model_path)
  result_test = best_model.evaluate(test, return_dict=True, verbose=0)
  print(result_test)

  # save predictions on the test set
  predictions = best_model.predict(test, verbose=0).numpy()
  np.save(f'./results/predictions/{model_name}', predictions)

if __name__ == '__main__':
  app.run(main)
