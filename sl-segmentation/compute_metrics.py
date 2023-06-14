import os
import sys
from args import FLAGS
from dataset import get_datasets
import numpy as np
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


# define prediction based on probabilities
def get_prediction(prob):
    pred = []
    for entry in prob:
        if entry[0] > entry[1]:
            pred.append(0)
        else:
            pred.append(1)
    return pred

FLAGS(sys.argv)

# read datasets
train, dev, test = get_datasets()

# get labels into lists
test_labels = []
for (input, label) in test:
    test_labels.append(label)

# list all results saved in folder
pred_files = sorted(glob('./results/predictions/*.npy'))

# compute metrics for each model
print('Test metrics:')
for file in pred_files:
    predictions = np.load(file, allow_pickle=True)
    # compute metrics
    acc = []
    prec = []
    rec = []
    size = []
    for i in range(predictions.shape[0]):
        # compute metrics for each video
        prob = np.array(predictions[i])
        pred_labels = np.array(get_prediction(prob))
        true_labels = test_labels[i].numpy()
        true_labels = np.reshape(true_labels, -1)
        acc.append(accuracy_score(true_labels, pred_labels))
        prec.append(precision_score(true_labels, pred_labels))
        rec.append(recall_score(true_labels, pred_labels))
        size.append(pred_labels.shape[0])
    # combine all videos using weighted average
    acc_total = np.average(acc, weights=size)
    prec_total = np.average(prec, weights=size)
    rec_total = np.average(rec, weights=size)

    model = file.split('/')[-1]
    print(f'\nModel: {model}')
    print(f'    Accuracy: {acc_total}')
    print(f'    Precision: {prec_total}')
    print(f'    Recall: {rec_total}')
    print(f'    F1 score: {(2*prec_total*rec_total)/(prec_total+rec_total)}')