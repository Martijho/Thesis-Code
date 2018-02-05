import sys
sys.path.append('../../')
from pathnet_keras import PathNet
from path_search import PathSearch, TS_box
from analytic import Analytic
from dataprep import DataPrep
from plot_pathnet import PathNetPlotter
import pickle as pkl
import time as clock
import numpy as np
import copy


class SearchExperiment:
    def __init__(self, data_list, hyperparameter_list, verbose=True):
        self.pn, task1 = PathNet.search_experiment(output_size=5, image_shape=[32, 32, 3])
        self.data = data_list
        self.hyperparameter_list = hyperparameter_list

        self.tasks = self._get_tasks(task1)
        self.verbose = verbose

    def _get_tasks(self, task1):

        task_config = task1.get_defining_config()

        task_config['name'] = 'unique_2'
        task_config['output'] = 5
        task2 = self.pn.create_new_task(config=task_config)

        task_config['name'] = 'unique_3'
        task_config['output'] = 10
        task3 = self.pn.create_new_task(config=task_config)

        task_config['name'] = 'unique_4'
        task_config['output'] = 5
        task4 = self.pn.create_new_task(config=task_config)

        task_config['name'] = 'unique_5'
        task_config['output'] = 5
        task5 = self.pn.create_new_task(config=task_config)

        task_config['name'] = 'unique_6'
        task_config['output'] = 10
        task6 = self.pn.create_new_task(config=task_config)

        return [task1, task2, task3, task4, task5, task6]

    def reset_object(self):
        self.pn, task1 = PathNet.search_experiment(output_size=5, image_shape=[32, 32, 3])
        self.tasks = self._get_tasks(task1)

    def run(self):
        ps = PathSearch(self.pn)

        paths       = []
        fitness     = []
        evaluated   = []
        training    = []
        generations = []

        for task, (x, y, x_t, y_t), param in zip(self.tasks, self.data, self.hyperparameter_list):

            p, f, log = ps.new_tournamet_search(x, y, task, hyperparam=param, verbose=self.verbose)
            training_counter =copy.deepcopy(self.pn.training_counter)

            self.pn.save_new_optimal_path(p, task)

            training.append(training_counter)
            paths.append(p)
            fitness.append(f)
            evaluated.append(self.pn.evaluate_path(x_t, y_t, p, task=task))
            generations.append(len(log['path']))

        return {'paths': paths,
                'fitness': fitness,
                'evaluated': evaluated,
                'training': training,
                'generations': generations}


def get_data_list():
    data = DataPrep()
    data.mnist()
    data.add_padding()
    data.grayscale2rgb()

    x1, y1, x_t1, y_t1 = data.sample_dataset([0, 1, 2, 3, 4])
    x2, y2, x_t2, y_t2 = data.sample_dataset([5, 6, 7, 8, 9])
    x3, y3, x_t3, y_t3 = data.sample_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    data = DataPrep()
    data.cSVHN_ez()

    x4, y4, x_t4, y_t4 = data.sample_dataset([0, 1, 2, 3, 4])
    x5, y5, x_t5, y_t5 = data.sample_dataset([5, 6, 7, 8, 9])
    x6, y6, x_t6, y_t6 = data.sample_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    return [(x1, y1, x_t1, y_t1), (x2, y2, x_t2, y_t2),
            (x3, y3, x_t3, y_t3), (x4, y4, x_t4, y_t4),
            (x5, y5, x_t5, y_t5), (x6, y6, x_t6, y_t6)]
def get_param_list():
    param1 = {'name': 'Task 1: MNIST [0, 1, 2, 3, 4]',
              'threshold_acc': 1.1,
              'generation_limit': 50,
              'population_size': 40,
              'selection_preassure': 3,
              'replace_func': TS_box.replace_3to2}

    param2 = dict(param1)
    param2['name'] = 'Task 2: MNIST [5, 6, 7, 8, 9]'

    param3 = dict(param1)
    param3['name'] = 'Task 3: MNIST [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'
    #param3['threshold_acc'] = 0.95

    param4 = dict(param1)
    param4['name'] = 'Task 4: cSVHN [0, 1, 2, 3, 4]'
    #param4['threshold_acc'] = 0.85

    param5 = dict(param1)
    param5['name'] = 'Task 5: cSVHN [5, 6, 7, 8, 9]'
    #param5['threshold_acc'] = 0.85

    param6 = dict(param1)
    param6['name'] = 'Task 6: cSVHN [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'
    #param6['threshold_acc'] = 0.80

    return [param1, param2, param3, param4, param5, param6]
def thread_task(thread_nr, data, param, experiments):
    try:
        with open('../../logs/search/thread_log_'+str(thread_nr)+'.pkl', 'rb') as file:
            log = pkl.load(file)
    except FileNotFoundError:
        log = {'path1':  [], 'path2':  [], 'path3':  [], 'path4':  [], 'path5':  [], 'path6':  [],
               'fit1':   [], 'fit2':   [], 'fit3':   [], 'fit4':   [], 'fit5':   [], 'fit6':   [],
               'eval1':  [], 'eval2':  [], 'eval3':  [], 'eval4':  [], 'eval5':  [], 'eval6':  [],
               'train1': [], 'train2': [], 'train3': [], 'train4': [], 'train5': [], 'train6': [],
               'gen1':   [], 'gen2':   [], 'gen3':   [], 'gen4':   [], 'gen5':   [], 'gen6':   []}

    experiment = SearchExperiment(data, param)

    for i in range(1, experiments + 1):
        experiment.reset_object()
        l = experiment.run()

        for j in range(len(param)): log['path' + str(j + 1)].append(l['paths'][j])
        for j in range(len(param)): log['fit' + str(j + 1)].append(l['fitness'][j])
        for j in range(len(param)): log['eval' + str(j + 1)].append(l['evaluated'][j])
        for j in range(len(param)): log['train' + str(j + 1)].append(l['training'][j])
        for j in range(len(param)): log['gen' + str(j + 1)].append(l['generations'][j])

        pn_plotter = PathNetPlotter(experiment.pn)
        pn_plotter.plot_paths(l['paths'], filename='iter'+str(i))

        print('\n\tFitness:\t', l['fitness'])
        print('\tEvaluated:\t', l['evaluated'])
        print('\tGenerations:\t', l['generations'], end='\n\n')

    with open('../../logs/search/thread_log_'+str(thread_nr)+'.pkl', 'wb') as f:
        pkl.dump(log, f)

if __name__ == "__main__":
    data  = get_data_list()
    param = get_param_list()

    thread_task(-1, data, param, 2)


