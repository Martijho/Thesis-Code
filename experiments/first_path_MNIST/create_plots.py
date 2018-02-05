import sys
sys.path.append('../')
from plotcreator import Exp1_plotter
import pickle
import numpy as np

log = None
dir = '../../../logs/first_path_MNIST/mnist_3layerConv_0.975'
with open(dir+'/log.pkl', 'rb') as file:
    log = pickle.load(file)

with open('task-flipped/log.pkl', 'rb') as file:
    f_log = pickle.load(file)

print('LOG WITH', len(log['s+s:path1']), 'EXPERIMENTS')
print('FLIPPED:', len(f_log['path1']), 'EXPERIMENTS')

ss = np.array([0, 0, 0])
ps = np.array([0, 0, 0])
fs = np.array([0, 0, 0])

for p1, p2 in zip(log['s+s:path1'], log['s+s:path2']):
    ss += np.array([len(x) for x in p1])
    ss += np.array([len(x) for x in p2])
ss = ss / (2 * len(log['s+s:path1']))

for p1, p2 in zip(log['p+s:path1'], log['p+s:path2']):
    ps += np.array([len(x) for x in p1])
    ps += np.array([len(x) for x in p2])
ps = ps / (2 * len(log['p+s:path1']))

for p1, p2 in zip(f_log['path1'], f_log['path2']):
    fs += np.array([len(x) for x in p1])
    fs += np.array([len(x) for x in p2])
fs = fs / (2 * len(f_log['path1']))

print('Avg ss: ', ss)
print('Avg ps: ', ps)
print('Avg fs: ', fs)

plotter = Exp1_plotter(log, 10, 3)
plotter.training_boxplot(lock=False, save_file=dir+'_training_boxplot', flipped=f_log)
plotter.reuse_barplot(lock=False, save_file=dir+'_module_reuse', flipped=f_log)
plotter.module_reuse_by_layer(lock=True, save_file=dir+'_reuse_by_layer', flipped=f_log)
#plotter.evaluation_vs_training(lock=True, save_file=dir+'_evaluation_vs_training')


