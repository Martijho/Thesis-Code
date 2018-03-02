import sys
sys.path.append('../../')
from plotcreator import Exp3_plotter
from analytic import Analytic
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join

def balance_log_sizes(logs):
    sizes = [len(l['path1']) for l in logs.values()]
    new_logs = {}
    for k, v in logs.items():
        new_v = {}
        for metric, data in v.items():
            new_v[metric] = data[:min(sizes)]
        new_logs[k] = new_v

    return new_logs
def print_reuse_table(table):
    print()
    print('   T1      T2      T3      T4      T5      T6  ')
    for i, row in enumerate(table):
        print('T' + str(i + 1) + '         ', end='        ' * i)
        for r in row:
            if r == 0.0:
                print(end='        ')
            else:
                print(str(round(r, 3)).ljust(5), end='   ')
        print()
    print('\n\n')

dir_path =  '../../../logs/overfit/'
log_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

logs = {}

for log_name in log_files:
    with open(dir_path+log_name, 'rb') as file:
        logs[log_name] = pickle.load(file)
    print(log_name.ljust(14), 'with', len(logs[log_name]), 'logged runs')

for k, v in logs.items():
    print(k)
    for metric in v[0].keys():
        print('\t', metric.ljust(50), end='')
        if len(str(v[0][metric])) > 50:
            print(str(v[0][metric])[:50], '...')
        else:
            print(v[0][metric])
    print()

for k, v in logs.items():
    for log in v:
        if log['static_transfer_evaluation_fitness_cSVHN'] == []:
            log['static_transfer_evaluation_fitness_cSVHN'] = log['static_transfer_evaluation_fitness_MNIST']
            log['static_transfer_evaluation_fitness_MNIST'] = -1
p = Exp3_plotter(logs)
p.plot_validation_accuracy()




