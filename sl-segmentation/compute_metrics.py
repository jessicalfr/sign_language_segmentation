import sys
from args import FLAGS
from dataset import get_datasets
import numpy as np
from glob import glob

# define metrics
def precision_score(y_true, y_pred):
    """
    Compute precision score given two 1D numpy arrays.
    """
    tp = np.multiply(y_true==y_pred, y_true)
    fp = np.multiply(y_true!=y_pred, y_pred)
    precision = sum(tp)/(sum(tp)+sum(fp))

    return precision

def recall_score(y_true, y_pred):
    """
    Compute recall score given two 1D numpy arrays.
    """
    tn = np.multiply(y_true==y_pred, (1-y_true))
    fp = np.multiply(y_true!=y_pred, y_pred)
    recall = sum(tn)/(sum(tn)+sum(fp))

    return recall

def accuracy_score(y_true, y_pred):
    """
    Compute accuracy score given two 1D numpy arrays.
    """
    return sum(y_true==y_pred)/len(y_true)


FLAGS(sys.argv)

# read datasets
train, dev, test = get_datasets()

# get labels into lists
dev_labels = []
for (input, label) in test:
    dev_labels.append(label)

print('Dev dataset:')
print(f'Dev length: {len(dev_labels)}')
print(f'Example label: {dev_labels[0]}')

test_labels = []
for (input, label) in test:
    test_labels.append(label)

print('Test dataset:')
print(f'Test length: {len(test_labels)}')
print(f'Example label: {test_labels[0]}')

# list all results saved in folder
predictions = glob('./results/predictions/*.npy')

print('.npy files:')
for file in predictions:
    print(f'   - {file}')

# for each model, get read probabilities, create prediction array (0 or 1), compute accuracy, precision and recall 