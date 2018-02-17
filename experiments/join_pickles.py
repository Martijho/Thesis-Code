from __future__ import print_function
import sys
import numpy as np
sys.path.append('../../')
import pickle as pkl

dir = '../../logs/first_path_MNIST/task-flipped/'
p35 = 'log_from_3.5.pkl'
p27 = 'log_from_2.7.pkl'

with open(dir + p27, 'rb') as file:
    log27 = pkl.load(file, fix_imports=True, encoding='bytes')

with open(dir + p35, 'rb') as file:
    log35 = pkl.load(file)


print(len(log27[b'path1']), 'experiments')
print(len(log35['path1']), 'experiments')

log = dict()
for k, v in log35.items():
    log[k] = log35[k] + log27[k.encode("utf-8")]

for k in log.keys(): print(k)
print(len(log['path1']))


with open(dir + 'log.pkl', 'wb') as f:
    pkl.dump(log, f)
