import sys
sys.path.append('../../')
from pathnet_keras import PathNet
from path_search import PathSearch
from analytic import Analytic
from dataprep import DataPrep
from plot_pathnet import PathNetPlotter
from datetime import datetime as dt
from matplotlib import pyplot as plt
import pickle as pkl
import os
import time as clock
import numpy as np


repeates            = 1
WRITE_LOG           = True
accuracy_threshold  = 0.975
noise               = False
search_hyper_param  = {'batch_size': 16,
                       'training_iterations': 50,
                       'population_size': 64}
dir_name            = '../../../logs/first_path_MNIST/task-flipped'

data = DataPrep()
data.mnist()

x1, y1, x_test1, y_test1 = data.sample_dataset([5, 6, 7, 8, 9])
x2, y2, x_test2, y_test2 = data.sample_dataset([0, 1, 2, 3, 4])

try:
    with open(dir_name+'/log.pkl', 'rb') as file:
        log = pkl.load(file)
except FileNotFoundError:
    log = {'path1':[], 'path2':[],
           'eval1':[], 'eval2':[],
           'gen1':[],  'gen2':[],
           'avg_training1':[],
           'avg_training2':[],
           'module_reuse':[]
           }

iteration = len(log['gen1'])
while True:
    iteration +=1
    START = clock.time()

    print('\n'*3, '\t'*3, 'ITERATION NR', iteration, '\n'*2)
    pn, first_task = PathNet.mnist(output_size=5)
    ps = PathSearch(pn)

    path1, fit1, log1 = ps.tournamet_search(x1, y1, first_task, stop_when_reached=accuracy_threshold, hyperparam=search_hyper_param)
    evaluation1 = pn.evaluate_path(x_test1, y_test1, path1, first_task)
    pn.save_new_optimal_path(path1, first_task)
    pn.reset_backend_session()

    second_task = pn.create_new_task(like_this=first_task)
    path2, fit2, log2 = ps.tournamet_search(x2, y2, second_task, stop_when_reached=accuracy_threshold, hyperparam=search_hyper_param)
    evaluation2 = pn.evaluate_path(x_test2, y_test2, path2, second_task)
    pn.save_new_optimal_path(path2, second_task)


    _, avg1 = Analytic.training_along_path(path1, pn.training_counter)
    _, avg2 = Analytic.training_along_path(path2, pn.training_counter)

    if WRITE_LOG:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2], filename=dir_name+'/itr'+str(iteration))
    else:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2])


    log['path1'].append(path1)
    log['path2'].append(path2)
    log['eval1'].append(evaluation1)
    log['eval2'].append(evaluation2)
    log['gen1'].append(len(log1['path']))
    log['gen2'].append(len(log2['path']))
    log['avg_training1'].append(avg1)
    log['avg_training2'].append(avg2)
    log['module_reuse'].append(Analytic.path_overlap(path1, path2))


    print('\tTask one:', 'Avg training:', '%.1f' % avg1, 'Fitness:', '%.4f' % evaluation1)
    print('\tTask two:', 'Avg training:', '%.1f' % avg2, 'Fitness:', '%.4f' % evaluation2)
    print('\tOverlap: ', log['module_reuse'][-1])



    STOP = clock.time()
    print('Experiment took', STOP-START, 'seconds')

    with open(dir_name + '/log.pkl', 'wb') as f:
        pkl.dump(log, f)

'''
for iteration in range(1, repeates+1):
    print('\n'*3, '\t'*3, 'ITERATION NR', iteration, '\n'*2)
    pn, first_task = PathNet.mnist(output_size=5)
    ps = PathSearch(pn)

    path1 = log['s+s:path1'][iteration-1]
    model = pn.path2model(path1, first_task)
    training_iterations = log['s+s:gen1'][iteration-1]
    hist = []
    print('Training random path:', path1, 'for', training_iterations,'x 50 mini batches')
    for training_iteration in range(training_iterations):
        for batch_nr in range(50):
            batch = np.random.randint(0, len(x1), 16)
            hist.append(model.train_on_batch(x1[batch], y1[batch])[1])
        pn.increment_training_counter(path1)
    evaluation1 = pn.evaluate_path(x_test1, y_test1, path1, first_task)
    pn.save_new_optimal_path(path1, first_task)

    second_task = pn.create_new_task(like_this=first_task)
    path2, fit2, log2 = ps.tournamet_search(x2, y2, second_task, stop_when_reached=accuracy_threshold, hyperparam=search_hyper_param)
    evaluation2 = pn.evaluate_path(x_test2, y_test2, path2, second_task)

    pn.save_new_optimal_path(path2, second_task)

    pn.reset_backend_session()

    print('Task one:', 'Avg training:', training_iterations, 'Fitness:', '%.4f' % evaluation1)
    print('Task two:', 'Avg training:', '%.1f' % avg2, 'Fitness:', '%.4f' % evaluation2)
    print('Overlap: ', Analytic.path_overlap(path1, path2))
    print('Overlap in s+s:', log['s+s:module_reuse'][iteration-1])

    log['p+s:path1'].append(path1)
    log['p+s:path2'].append(path2)
    log['p+s:eval1'].append(evaluation1)
    log['p+s:eval2'].append(evaluation2)
    log['p+s:gen1'].append(training_iterations)
    log['p+s:gen2'].append(len(log2['path']))
    _, avg1 = Analytic.training_along_path(path1, pn.training_counter)
    log['p+s:avg_training1'].append(avg1)
    _, avg2 = Analytic.training_along_path(path2, pn.training_counter)
    log['p+s:avg_training2'].append(avg2)
    log['p+s:module_reuse'].append(Analytic.path_overlap(path1, path2))

    if WRITE_LOG:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2], filename=now+'/p+s:itr'+str(iteration)+':Reuse'+str(log['s+s:module_reuse'][-1]))
    else:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2])
'''
