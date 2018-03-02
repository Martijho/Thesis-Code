import sys
sys.path.append('../../')
from plotcreator import Exp2_plotter
from analytic import Analytic
import pickle
import numpy as np

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

dir_path =  '../../../logs/search/'

experiments = {'low2high': {}, 'high2low': {}, 'low': {}, 'recomb':{}, 'high': {}}
log_files = {'low2high': 'log_l2h.pkl',
             'high2low': 'log_h2l.pkl',
             'high': 'log_h.pkl',
             'low': 'log_l.pkl',
             'recomb':'log_recomb.pkl'}

for exp, log_name in log_files.items():
    try:
        with open(dir_path+log_name, 'rb') as file:
            experiments[exp] = pickle.load(file)
        print(exp.ljust(10), 'with', len(experiments[exp]['path1']), 'logged runs')
    except:
        print(exp.ljust(10), 'no log')
        del experiments[exp]

experiments = balance_log_sizes(experiments)
for k, v in experiments.items():
    print(k, len(list(v.values())[0]))

log = experiments['low2high']
printed = []
for k, v in log.items():
    if k[:-1] not in printed:
        #print(k[:-1])
        pass
    printed.append(k[:-1])

p = Exp2_plotter(experiments)

p.plot_average_population_diversity(save_file='plots/Average_population_diversity_homemade',
                                    diversity_metric=Analytic.homemade_diversity, lock=False)
input()
p.boxplot_used_training_ratio(save_file='plots/Used_training_ratio', lock=False)
#for k, v in experiments.items():
#    p.plot_population_diversity(v, k, save_file='plots/Population_diversity_'+k, lock=False)
p.plot_capacity_vs_fitness(save_file='plots/Capacity_pr_validation_accuracy', lock=False)
p.plot_training_progress(save_file='plots/Training_accuracy', lock=False)
p.plot_capacity_usage(save_file='plots/Capacity_pr_generation', lock=False)
p.plot_training_iterations(save_file='plots/Training_value', lock=False)
p.plot_reuse(save_file='plots/Module_reuse_pr_generation', lock=False)
p.plot_average_path_size(save_file='plots/Average_path_size', lock=False)
p.plot_layer_sizes(save_file='plots/Average_size_by_layer', lock=False)

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





