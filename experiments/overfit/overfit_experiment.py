from __future__ import print_function
import sys
sys.path.append('../../')
from pathnet_keras import PathNet
from path_search import PathSearch, TS_box
from analytic import Analytic
from dataprep import DataPrep
import pickle as pkl
import time
import numpy as np
import copy
import threading
from keras import backend as K
import tensorflow as tf
import sys
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.merge import add
from keras.optimizers import Adagrad, Adam, RMSprop, SGD

def print_sizes(data):
    print('Total Training:', len(data.x), '\tTotal validation:', len(data.x_test))
    for i in range(10):
        x, y, x_t, y_t = data.sample_dataset([i])
        print('\t Class:', i, '\tTraining:', len(x), '\tValidation:', len(x_t))
    print()
def get_data(limit, validation_size):
    data = DataPrep()
    data.cSVHN_ez(validation_split=0.7)
    data.limit_data(training=limit, validation=validation_size, balance=True)
    return data.x, data.y, data.x_test, data.y_test
def create_empty_log():
    log_ = {'pathnet_path': [],
            'pathnet_training': [],
            'pathnet_log': [],
            'pathnet_eval': [],
            'pathnet_fitness': [],
            'pathnet_size': [],
            'pathnet_reuse': [],
            'pathnet_denovo_path': [],
            'pathnet_denovo_training': [],
            'pathnet_denovo_log': [],
            'pathnet_denovo_eval': [],
            'pathnet_denovo_fitness': [],
            'pathnet_denovo_size': [],
            'static_training_fitness': [],
            'static_evaluation_fitness': [],
            'static_size': [],
            'static_transfer_training_fitness_MNIST': [],
            'static_transfer_evaluation_fitness_MNIST': [],
            'static_transfer_size_MNIST': [],
            'static_transfer_training_fitness_cSVHN': [],
            'static_transfer_evaluation_fitness_cSVHN': [],
            'static_transfer_size_cSVHN': []
            }
    return log_

data = DataPrep()
data.mnist()
data.add_padding()
data.grayscale2rgb()
x1, y1, x1_test, y1_test = data.x, data.y, data.x_test, data.y_test


K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=4))))
t = time.time()
for limit in [40000, 20000, 10000, 5000, 1000]:
    limit/=10
    x2, y2, x2_test, y2_test = get_data(limit, 100000)
    print()
    print('========== ********* =========== Training size:', limit, '========== ********* ===========')
    print('       Training:', x2.shape, 'Validation:', x2_test.shape)
    print()

    #x1_test, y1_test = x1_test[:100], y1_test[:100]
    log = create_empty_log()

    # De Novo PathNet
    pn, task1 = PathNet.overfit_experiment()
    ps = PathSearch(pn)                 
    param = {'name': 'Task 1: cSVHN',
              'threshold_acc': 1.1,
              'generation_limit': 100, #80
              'population_size': 64,
              'selection_pressure': [2],#2,
              'replace_func': [TS_box.winner_replace_all],
              'flipped': False}
    #denovo_path, denovo_training_fitness, denovo_log = ps.new_tournamet_search(x2, y2, task1, hyperparam=param)
    denovo_path, denovo_training_fitness, denovo_log = ps.dynamic_tournamet_search(x2, y2, task1, hyperparam=param)

    log['hyperparameter'] = param

    log['pathnet_denovo_path'] = denovo_path
    log['pathnet_denovo_fitness'] = denovo_training_fitness
    log['pathnet_denovo_size'] = pn.path2model(denovo_path, task1).count_params()
    log['pathnet_denovo_log'] = denovo_log
    log['pathnet_denovo_training'] = copy.deepcopy(pn.training_counter)
    log['pathnet_denovo_eval'] = pn.evaluate_path(x2_test, y2_test, path=denovo_path, task=task1)
    _, PN_E = Analytic.training_along_path(denovo_path, pn.training_counter)
    print(denovo_path, log['pathnet_denovo_eval'])



    # Standard multitask PathNet
    pn, task2 = PathNet.overfit_experiment()
    ps = PathSearch(pn)

    param['name'] = 'Task 1: MNIST'
    #mnist_path, mnist_training_fitness, mnist_log = ps.new_tournamet_search(x1, y1, task2, hyperparam=param)
    mnist_path, mnist_training_fitness, mnist_log = ps.dynamic_tournamet_search(x1, y1, task2, hyperparam=param)
    pn.save_new_optimal_path(mnist_path, task2)
    _, PN_Pmnist = Analytic.training_along_path(mnist_path, pn.training_counter)

    task3 = pn.create_new_task(like_this=task2)
    param['name'] = 'Task 2: cSVHN'
    #cSVHN_path, cSVHN_training_fitness, cSVHN_log = ps.new_tournamet_search(x2, y2, task3, hyperparam=param)
    cSVHN_path, cSVHN_training_fitness, cSVHN_log = ps.dynamic_tournamet_search(x2, y2, task3, hyperparam=param)
    log['pathnet_path'] = cSVHN_path
    log['pathnet_fitness'] = cSVHN_training_fitness
    log['pathnet_size'] = pn.path2model(cSVHN_path, task3).count_params()
    log['pathnet__log'] = cSVHN_log
    log['pathnet_training'] = copy.deepcopy(pn.training_counter)
    log['pathnet_eval'] = pn.evaluate_path(x2_test, y2_test, path=cSVHN_path, task=task3)
    log['pathnet_reuse'] = Analytic.path_overlap(mnist_path, cSVHN_path)
    _, PN_P = Analytic.training_along_path(cSVHN_path, pn.training_counter)
    print('MNIST:', mnist_path, pn.evaluate_path(x1_test, y1_test, path=mnist_path, task=task2))
    print('cSVHN:', cSVHN_path, log['pathnet_eval'])
    print('overlap:', Analytic.path_overlap(mnist_path, cSVHN_path))


    # Static ML
    pn, task1 = PathNet.overfit_experiment()
    model = pn.path2model(denovo_path, task1)
    static_ML_training_acc = []
    for i in range(int(round(PN_E))*50):
        batch = np.random.randint(0, len(x2), 16)
        hist = model.train_on_batch(x2[batch], y2[batch])
        static_ML_training_acc.append(hist[1])

    fit = model.evaluate(x2_test, y2_test, batch_size=16, verbose=False)[1]
    log['static_training_fitness'] = static_ML_training_acc
    log['static_evaluation_fitness'] = fit
    log['static_size'] = model.count_params()
    print('Static ML:', '50*'+str(PN_E), 'iterations. Reached', fit)

    # TL in static ML
    pn, task2 = PathNet.overfit_experiment()
    model = pn.path2model(cSVHN_path, task2)
    ML_MNIST_training_acc = []
    ML_cSVHN_training_acc = []
    for i in range(int(round(PN_Pmnist))*50):
        batch = np.random.randint(0, len(x1), 16)
        hist = model.train_on_batch(x1[batch], y1[batch])
        ML_MNIST_training_acc.append(hist[1])

    fit1 = model.evaluate(x1_test, y1_test, batch_size=16, verbose=False)[1]
    print('Transfer learning MNIST:', '50*'+str(PN_Pmnist), 'iterations. Reached', fit1)

    for i in range(int(round(PN_P))*50):
        batch = np.random.randint(0, len(x2), 16)
        hist = model.train_on_batch(x2[batch], y2[batch])
        ML_cSVHN_training_acc.append(hist[1])
    fit2 = model.evaluate(x2_test, y2_test, batch_size=16, verbose=False)[1]
    print('Transfer learning cSVHN:', '50*'+str(PN_P), 'iterations. Reached', fit2)

    log['static_transfer_training_fitness_MNIST'] = ML_MNIST_training_acc
    log['static_transfer_evaluation_fitness_MNIST'] = fit1
    log['static_transfer_size_MNIST'] = model.count_params()
    log['static_transfer_training_fitness_MNIST'] = ML_cSVHN_training_acc
    log['static_transfer_evaluation_fitness_MNIST'] = fit2
    log['static_transfer_size_MNIST'] = model.count_params()
    log['set_size'] = limit

    log['PN_E'] = PN_E
    log['PN_Pmnist'] = PN_Pmnist
    log['PN_P'] = PN_P

    try:
        with open('../../../logs/overfit/log'+str(limit)+'.pkl', 'rb') as file:
            log_list = pkl.load(file)
        log_list.append(log)
    except IOError:
        log_list = [log]

    with open('../../../logs/overfit/log'+str(limit)+'.pkl', 'wb') as f:
        pkl.dump(log_list, f)
print('TOTAL RUNTIME:', round((time.time()-t)/60, 3), 'min')




