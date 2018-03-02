from __future__ import print_function
import sys
import numpy as np
sys.path.append('../../')
import pickle as pkl


def convert_dict_keys(data):
    tmp = dict()
    for tk, tv in data.items():
        if type(tv) is dict:
            tv = convert_dict_keys(tv)
        elif type(tv) is list and type(tv[0]) is dict:
            tv = [convert_dict_keys(x) for x in tv]
        tmp[tk.decode('utf-8')] = tv
    return tmp

dir = '../../logs/search/'
#p35 = 'log_from_3.5.pkl'
#p27 = 'log_from_2.7.pkl'


log_names = ['log_h.pkl', 'log_h_2.pkl','log_h_3.pkl',
             'log_l.pkl', 'log_l_3.pkl',
             'log_l2h.pkl', 'log_l2h_2.pkl',
             'log_h2l.pkl', 'log_h2l_2.pkl',
             'log_recomb.pkl', 'log_recomb_2.pkl']

logs = {}
for log_name in log_names:
    try:
        with open(dir +'dunder_logs/'+ log_name, 'rb') as file:
            log = pkl.load(file, fix_imports=True, encoding='bytes')
            print('\t\tReading', log_name, 'with', len(list(log.values())[0]))
    except:
        print('\t\t' + log_name, 'DOES NOT EXIST')
        continue

    if '_2.' in log_name or '_3.' in log_name or '_4.' in log_name:
        log_name = log_name[:-6] + log_name[-4:]
    if log_name not in logs.keys():
        logs[log_name] = log
    else:
        print(log_name, ' joining:', len(logs[log_name][b'eval1']), '+', len(log[b'eval1']))
        for k, v in logs[log_name].items():
            v += log[k]

print()
new = {}
for k, v in logs.items():
    new[k] = convert_dict_keys(v)
logs = new

for k, v in logs.items():
    print('Log:', k)
    print('Keys:', len(v.keys()))
    print('Runs:', len(list(v.values())[0]))
    print('log-keys:', list(v['log1'][0].keys()))
    print()


for k, v in logs.items():
    with open(dir +'dunder_logs/joined_files/' + k, 'wb') as f:
        pkl.dump(new, f)

log_names = ['log_h.pkl', 'log_l.pkl', 'log_recomb.pkl', 'log_l2h.pkl', 'log_h2l.pkl']

for log_name in log_names:
    with open(dir + 'laptop_logs/' + log_name, 'rb') as f:
        log = pkl.load(f)
        print('\t\tReading', log_name, 'with', len(list(log.values())[0]))

    if log_name in logs.keys():
        for k, v in log.items():
            logs[log_name][k] += v

    else:
        logs[log_name] = log


for k, v in logs.items():
    print('Log:', k)
    print('Keys:', len(v.keys()))
    print('Runs:', len(list(v.values())[0]))
    print('log-keys:', list(v['log1'][0].keys()))
    print()



for k, v in logs.items():
    print(k, len(v['eval1']))
    with open(dir +  k, 'wb') as f:
        pkl.dump(v, f)



'''


log_names = ['log_h.pkl', 'log_l.pkl', 'log_recomb.pkl', 'log_l2h.pkl', 'log_h2l.pkl']
path_suffixes = []
for x in list(log_names):
    path_suffixes.append('dunder_logs/joined_files/' + x)
    path_suffixes.append('laptop_logs/' + x)

log_names = path_suffixes
logs = {}
for log_name in log_names:
    try:
        with open(dir + log_name, 'rb') as file:
            log = pkl.load(file)
    except: continue
    if 'dunder' in log_name:
        log_name = log_name[25:]
    if 'laptop' in log_name:
        log_name = log_name[12:]

    try: log = convert_dict_keys(log)
    except: pass

    if log_name not in logs.keys():
        logs[log_name] = log
    else:
        for k, v in logs[log_name].items():
            print(k)
            v += log[k]

for k, v in logs.items():
    print(k, len(v['eval1']))
    with open(dir +  k, 'wb') as f:
        pkl.dump(v, f)
'''
'''

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
'''