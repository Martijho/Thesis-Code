import sys
sys.path.append('../../')
from plotcreator import Exp2_plotter
import pickle
import numpy as np

def print_reuse_table(table):
    print('   T1    T2    T3    T4    T5    T6  ')
    for i, row in enumerate(table):
        print('T' + str(i + 1) + '       ', end='      ' * i)
        for r in row:
            if r == 0.0:
                print(end='      ')
            else:
                print(r, end='   ')
        print()
    print('\n\n')

dir_path =  '../../../logs/search/'

experiments = {'low2high': {}, 'high2low': {}, 'high': {}, 'low': {} }
log_files = {'low2high': 'log_l2h.pkl',
             'high2low': 'log_h2l.pkl',
             'high': 'log_h.pkl',
             'low': 'log_l.pkl'}

for exp, log_name in log_files.items():
    with open(dir_path+log_name, 'rb') as file:
        experiments[exp] = pickle.load(file)
    print(exp.ljust(10), 'with', len(experiments[exp]['path1']), 'logged runs')

log = experiments['low2high']
for k, v in log.items():
    print(k, v)


'''
# printing average reuse tables for all experiments
for k, v in experiments.items():
    if len(v['path1']) == 1: continue
    table, avg_reuse = Exp2_plotter.average_reuse(v)
    print(k+':', avg_reuse)
    print_reuse_table(table)


# printing average reuse tables for all experiments
for k, v in experiments.items():
    print(k)
    usefull_training = Exp2_plotter.usefull_training_pr_task(v)
    for task, ratio in usefull_training.items():
        print(('\t> ' + task+': '+str(sum(ratio)/len(ratio))).ljust(40), ratio)
    print()
'''
plotter = Exp2_plotter(experiments)
plotter.boxplot_used_training_ratio()

