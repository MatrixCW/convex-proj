# -*- coding: utf-8 -*-
"""Sample user program that use PS.

"""

import numpy as np
import scipy.io
from parameter_server import param_store
from parameter_server import postoffice
from parameter_server import consistency
from parameter_server import ps_main

TRAIN_FEATURE_FILE = './train_features.mat'
TRAIN_LABEL_FILE = './train_labels.mat'

LAM = 1.0
RATE = 0.005
C = 10
feature_cnt = 401  # each data point has 401 features

param_store.add_param('w', np.zeros(C * feature_cnt))
param_store.add_param('b', np.zeros(C))


def worker_main(worker_id):
    postoffice.init_sender(worker_id)

    params = postoffice.pull_params([('w', 0, 20), ('b', 0, 10)], worker_id)
    
    print(params)
    
    # do computation

    postoffice.push_params(new_params_from_computation)

    # sync according to consistency model
    postoffice.sync()


ps_main.run_ps(worker_main, consistency.SEQUENTIAL, 1, 0.5)
