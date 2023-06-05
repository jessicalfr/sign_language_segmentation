## Execute all experiments

import random
import pandas as pd
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

random.seed(169)

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# create folders
make_dir('results')
make_dir('results/logs')
make_dir('results/models')
make_dir('results/plots')
make_dir('results/predictions')

# list all hiperparameters
hyperparameters = {
    'encoder_bidirectional': [True, False],
    #'encoder_layers': [1, 2, 3, 4, 5],
    'hidden_size': [32, 64, 128],
    'input_components': ['only pose', 'only hands', 'pose, face and hands'],
    'learning_rate': [0.01, 0.001, 0.0001],
    'input_dropout': [0.1, 0.2, 0.3, 0.4, 0.5] # random feature dropout
}

# random grid search
num_experiments = 30
experiment_def = []

for i in range(num_experiments):
    flags_string = ''
    for param in hyperparameters.keys():
        value = random.sample(hyperparameters[param],1)[0]
        if param == 'input_components':
            if value == 'only pose':
                flags_string = flags_string + ' --input_components pose_keypoints_2d'
            elif value == 'only hands':
                flags_string = flags_string + ' --input_components hand_left_keypoints_2d --input_components hand_right_keypoints_2d'
            else:
                flags_string = flags_string + ' --input_components pose_keypoints_2d --input_components face_keypoints_2d --input_components hand_left_keypoints_2d --input_components hand_right_keypoints_2d'
        elif param == 'encoder_bidirectional':
            if value:
                flags_string = flags_string + ' --encoder_bidirectional' # False is the default value defined in args.py
        else:
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