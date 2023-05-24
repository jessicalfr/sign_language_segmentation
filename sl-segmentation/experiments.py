## Execute all experiments

import random
import pandas as pd
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# list all hiperparameters
hyperparameters = {
    'encoder_bidirectional': [True, False],
    #'encoder_layers': [1, 2, 3, 4, 5],
    'hidden_size': [32, 64, 128, 256],
    'input_components': ['pose_keypoints_2d', # only pose
                         'hand_left_keypoints_2d, hand_right_keypoints_2d', # only hands
                         'pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d'], # pose, face and hands
    'learning_rate': [0.01, 0.001, 0.0001],
    'input_dropout': [0.1, 0.3, 0.5] # random feature dropout
}

# random grid search
num_experiments = 2
experiment_def = []

for i in range(num_experiments):
    flags_string = ''
    for param in hyperparameters.keys():
        value = random.sample(hyperparameters[param],1)[0]
        flags_string = flags_string + ' --' + param + ' ' + str(value)
    experiment_def.append(flags_string)

# save experiments metadata
metadata = {'experiment_number': list(range(num_experiments)),
            'flags': experiment_def}

metadata_pd = pd.DataFrame(metadata)
metadata_path = 'results/experiments_metadata.csv'
metadata_pd.to_csv(metadata_path)
print(f'Experiments metadata saved in {metadata_path}\n')

# execute training
for i in range(num_experiments):
    start = time.time()
    print(f'Executing experiment #{i} ...')
    train_string = f'python3 train.py {experiment_def[i]} --model_path results/models/model_{i}.h5 --dataset_path data/dgs_corpus.tfrecord > results/logs/experiment_{i}.log'
    os.system(train_string)
    end = time.time()
    print(f'Completed in {round(end-start)}s.')
    print(f'Log saved in results/logs/experiment_{i}.log')
    print(f'Best model saved in results/models/model_{i}.h5')
    print('')