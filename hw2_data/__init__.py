import numpy as np
import scipy.io as sio
import os
from os.path import join as pathjoin

curr_dir = os.path.dirname(os.path.realpath(__file__))


def load():
    if not os.path.isfile(pathjoin(curr_dir, 'hw2.mat')):
        train_features = np.genfromtxt(pathjoin(curr_dir, 'train_features.csv'), delimiter=',')
        train_labels = np.genfromtxt(pathjoin(curr_dir, 'train_labels.csv'), delimiter=',')
        test_features = np.genfromtxt(pathjoin(curr_dir, 'test_features.csv'), delimiter=',')
        test_labels = np.genfromtxt(pathjoin(curr_dir, 'test_labels.csv'), delimiter=',')
        sio.savemat(pathjoin(curr_dir, 'hw2.mat'), {
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels
        })
    mat = sio.loadmat(pathjoin(curr_dir, 'hw2.mat'))
    return np.matrix(mat['train_features']), np.matrix(mat['train_labels'].astype('int')).T
