import sys
sys.path.append('../')
from datetime import datetime as dt
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from analytic import Analytic
import numpy as np
import random

# First path experiments + Flipped Search+Search
class Exp1_plotter:
    def __init__(self, log, width, depth):
        self.log = log
        self.width = width
        self.depth = depth
        self.number_of_experiments = len(log['s+s:path1'])

        self.ss_color = '#1f77b4'
        self.ps_color = '#ff7f0e'
        self.random_color ='#2ca02c'

        self.max_reuse = max(self.log['s+s:module_reuse']+self.log['p+s:module_reuse'])

        # Probabilities of index number of overlap between two randomly chosen models
        # given width = 10, depth = 3 and uniformly chosen number of active modules between 1 and 3 from
        # each layer
        self.overlap_prob = [0.2604032750, 0.3960797996, 0.2457475490, 0.0806327611, 0.0152829566,
                             0.0014405546, 0.0001172514, 0.0000045799, 0.0000000945, 0.0000000008]

        self.module_reuse_prob = [0.6385, 0.32376, 0.03672, 0.00092]

        print('Stored metrics in log: ')
        for k, v in log.items():
            if 's+s' in k:
                print('  >', k[4:])

    def training_boxplot(self, save_file=None, lock=True, flipped=None):
        if flipped is not None:
            self.max_reuse = max([self.max_reuse] + flipped['module_reuse'])

        def draw_plot(data, offset, edge_color, fill_color):
            pos = np.arange(self.max_reuse+1)+offset
            bp = ax.boxplot(data, positions=pos, widths=0.2, patch_artist=True, manage_xticks=False)
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)
            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

        ssbox = [[] for _ in range(self.max_reuse+1)]
        psbox = [[] for _ in range(self.max_reuse+1)]

        for ss_avg, ss_mr, ps_avg, ps_mr in zip(self.log['s+s:avg_training2'], self.log['s+s:module_reuse'],
                                                self.log['p+s:avg_training2'], self.log['p+s:module_reuse']):
            ssbox[ss_mr].append(ss_avg)
            psbox[ps_mr].append(ps_avg)


        fig, ax = plt.subplots()
        plt.title('Average training vs Module reuse')
        draw_plot(ssbox, -0.1, self.ss_color, "white")
        draw_plot(psbox, +0.1, self.ps_color, "white")

        if flipped is not None:
            fssbox = [[] for _ in range(self.max_reuse + 1)]

            for reuse, avg in zip(flipped['module_reuse'], flipped['avg_training2']):
                fssbox[reuse].append(avg)

            draw_plot(fssbox, 0, 'red', 'white')

        plt.xticks(range(self.max_reuse+1))
        plt.ylabel('Avg training')
        plt.xlabel('Module reuse')
        ss_patch = mpatches.Patch(color=self.ss_color, label='Search + Search')
        ps_patch = mpatches.Patch(color=self.ps_color, label='Pick + Search')
        handles = [ss_patch, ps_patch]

        for i in range(len(ssbox)):
            if len(ssbox[i]) != 0: ssbox[i] = sum(ssbox[i]) / len(ssbox[i])
            #else:                  ssbox[i] = 0
            if len(psbox[i]) != 0: psbox[i] = sum(psbox[i]) / len(psbox[i])
            #else:                  psbox[i] = 0
        ssbox.pop(-1)
        psbox.pop(-1)
        plt.plot(ssbox, color=self.ss_color)
        plt.plot(psbox, color=self.ps_color)

        if flipped is not None:
            handles.append(mpatches.Patch(color='red', label='Search + Search flipped'))
            for i in range(len(fssbox)):
                if len(fssbox[i]) != 0: fssbox[i] = sum(fssbox[i]) / len(fssbox[i])
                else:                   fssbox[i] = 0
            plt.plot(fssbox, color='red')

        plt.legend(handles=handles)


        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)

    def reuse_barplot(self, save_file, lock=True, flipped=None):
        plt.figure('Module reuse')
        plt.title('Module reuse')

        max_reuse = max(self.log['p+s:module_reuse'] + self.log['s+s:module_reuse'])
        if flipped is not None:
            max_reuse = max([max_reuse] + flipped['module_reuse'])

        ss_reuse  = [0] * (max_reuse + 1)
        ps_reuse  = [0] * (max_reuse + 1)
        ssf_reuse = [0] * (max_reuse + 1)
        rnd_reuse = self.overlap_prob[:max_reuse+1]

        for x in self.log['s+s:module_reuse']: ss_reuse[x]+=1
        for x in self.log['p+s:module_reuse']: ps_reuse[x]+=1

        if flipped is not None:
            for x in flipped['module_reuse']: ssf_reuse[x]+=1

        for i in range(max_reuse+1):
            ss_reuse[i]  /= len(self.log['s+s:module_reuse'])
            ps_reuse[i]  /= len(self.log['p+s:module_reuse'])
            if flipped is not None:
                ssf_reuse[i] /= len(flipped['module_reuse'])

        ind = np.arange(max_reuse+1)
        width = 0.15

        if flipped is not None:
            plt.bar(ind - 2 * width, ss_reuse, width, color=self.ss_color)
            plt.bar(ind - width, ps_reuse, width, color=self.ps_color)
            plt.bar(ind, ssf_reuse, width, color='red')
            plt.bar(ind + width, rnd_reuse, width, color=self.random_color)

            plt.ylabel('% of experiments')
            plt.xticks(ind+width/4, list(range(max_reuse+1)))

            plt.legend(['S+S', 'P+S', 'S+S flipped', 'Random module selection'])
        else:

            plt.bar(ind - width, ss_reuse, width, color=self.ss_color)
            plt.bar(ind, ps_reuse, width, color=self.ps_color)
            plt.bar(ind + width, rnd_reuse, width, color=self.random_color)

            plt.ylabel('Reuse')
            plt.xticks(ind + width / 3, list(range(max_reuse + 1)))

            plt.legend(['S+S', 'P+S', 'Random module selection'])

        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)

    def module_reuse_by_layer(self, save_file=None, lock=True, flipped=None):
        plt.figure('Module reuse for each layer')
        plt.title('Module reuse for each layer')

        ss = []
        ps = []
        fs = []

        for i, (ss1, ss2, ps1, ps2) in enumerate(zip(self.log['s+s:path1'], self.log['s+s:path2'], self.log['p+s:path1'], self.log['p+s:path2'])):

            _ss_reuse = [0] * self.depth
            _ps_reuse = [0] * self.depth

            for j in range(self.depth):
                for m in ss2[j]:
                    if m in ss1[j]:
                        _ss_reuse[j]+=1
                for m in ps2[j]:
                    if m in ps1[j]:
                        _ps_reuse[j]+=1

            '''
            for j in range(self.depth):
                if sum(_ss_reuse) != 0: _ss_reuse[j] /= sum(_ss_reuse)
                if sum(_ps_reuse) != 0: _ps_reuse[j] /= sum(_ps_reuse)
            '''
            ss.append(_ss_reuse)
            ps.append(_ps_reuse)

        if flipped is not None:
            for i, (p1, p2) in enumerate(zip(flipped['path1'], flipped['path2'])):
                _reuse = [0]*self.depth

                for j in range(self.depth):
                    for m in p2[j]:
                        if m in p1[j]:
                            _reuse[j]+=1

                assert sum(_reuse) == flipped['module_reuse'][i], 'Error(flipped): counted reuse other than logged'

                #for j in range(self.depth):
                #    if sum(_reuse) != 0: _reuse[j] /= sum(_reuse)

                fs.append(_reuse)

        ss_mean = np.sum(np.array(ss), axis=0) / np.sum(np.array(ss))
        ps_mean = np.sum(np.array(ps), axis=0) / np.sum(np.array(ps))
        if flipped is not None: fs_mean = np.sum(np.array(fs), axis=0) / np.sum(np.array(fs))

        x = list(range(1, self.depth + 1))

        plt.scatter(x, ss_mean, label='S+S', color=self.ss_color)
        plt.scatter(x, ps_mean, label='P+S', color=self.ps_color)
        if flipped is not None:plt.scatter(x, fs_mean, label='S+S flipped', color='red')

        max_val = max(ss_mean.tolist() + ps_mean.tolist())
        if flipped is not None: max_val = max([max_val] + fs_mean.tolist())

        plt.legend()
        plt.xlabel('Layer')
        plt.ylabel('Module reuse')
        plt.ylim([0, max_val+0.1])
        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)


    def evaluation_vs_training(self, save_file=None, lock=True):
        #plt.figure('Model evaluation by avg training')
        #plt.title('Evaluation as function of training')
        max_training = max(self.log['s+s:avg_training1']+self.log['s+s:avg_training2']+self.log['p+s:avg_training1']+self.log['p+s:avg_training2'])

        f, axarr = plt.subplots(2, 2)

        self._eval_vs_training_subploter(axarr[0, 0], self.log['s+s:avg_training1'], self.log['s+s:eval1'],
                                         self.ss_color, 'o', 'S+S: Task 1')
        self._eval_vs_training_subploter(axarr[0, 1], self.log['s+s:avg_training2'], self.log['s+s:eval2'],
                                         self.ss_color, 'x', 'S+S: Task 2')
        self._eval_vs_training_subploter(axarr[1, 0], self.log['p+s:avg_training1'], self.log['p+s:eval1'],
                                         self.ps_color, 'o', 'P+S: Task 1')
        self._eval_vs_training_subploter(axarr[1, 1], self.log['p+s:avg_training2'], self.log['p+s:eval2'],
                                         self.ps_color, 'x', 'P+S: Task 2')


        #plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)



        #plt.scatter(self.log['s+s:avg_training1'], self.log['s+s:eval1'], color=self.ss_color, marker='o')
        #plt.scatter(self.log['s+s:avg_training2'], self.log['s+s:eval2'], color=self.ss_color, marker='x')
        #plt.scatter(self.log['p+s:avg_training1'], self.log['p+s:eval1'], color=self.ps_color, marker='o')
        #plt.scatter(self.log['p+s:avg_training2'], self.log['p+s:eval2'], color=self.ps_color, marker='x')

        #plt.legend(['s+s: Task 1', 's+s: Task 2', 'p+s: Task 1', 'p+s: Task 2'])
        #plt.xlabel('average training')
        #plt.ylabel('evaluation accuracy')


        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)
    def _eval_vs_training_subploter(self, subplot, training, eval, color, marker, title):
        training = np.array(training)
        eval = np.array(eval)

        none_outliers = np.array(eval) > 0.85
        training = training[none_outliers]
        eval = eval[none_outliers]

        subplot.scatter(training, eval, color=color, marker=marker)

        fit1 = np.polyfit(training, eval, 1)

        x = np.linspace(0, max(training), 1000)
        subplot.legend([title])
        subplot.plot(x, fit1[0] * x + fit1[1], color='red')
        #subplot.set_title(title + '['+str(round(fit1[0], 8)) +']')

    '''
    def module_reuse_histogram(self, save_file=None, lock=True, flipped=None):
        plt.figure('Module reuse')
        plt.title('Module reuse histogram')

        random_selection_overlap = []
        for i, P in enumerate(self.overlap_prob):
            random_selection_overlap+=int(round(self.number_of_experiments*P))*[i]

        while len(random_selection_overlap) < len(self.log['s+s:module_reuse']):
            random_selection_overlap.append(1)


        data = np.vstack([self.log['s+s:module_reuse'], self.log['p+s:module_reuse'], random_selection_overlap]).T

        bins = np.linspace(0, max(random_selection_overlap + self.log['p+s:module_reuse'] + self.log['s+s:module_reuse'] + flipped['module_reuse']), 16)

        print(len(flipped['module_reuse']))
        #n, bin, patches = plt.hist(data, 4*self.max_reuse)

        print(len(n))
        print(n[0])
        n[-1] = 250
        #print(bin)
        #print(patches)
        #print(bins)

        data = [self.log['s+s:module_reuse'], self.log['p+s:module_reuse'], random_selection_overlap, flipped['module_reuse']]
        plt.hist(data, bins, alpha=0.7, label=['Search + Search', 'Pick + Search', 'Overlap in random module selecton', 's+s flipped'],
                 color=[self.ss_color, self.ps_color, self.random_color, 'red'], density=True)

        plt.legend(loc='upper right')
        plt.xlabel('Module reuse')
        plt.ylabel('Frequency')
        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)
    '''

# Search experiments
class Exp2_plotter:
    def __init__(self, logs, number_of_tasks=6):
        self.logs = logs
        self.number_of_tasks = number_of_tasks
        self.colors = ['#dd2020', '#e8bd00', '#7b6500', '#0fcf71', '#409edb', '#e55d82']

    @staticmethod
    def average_reuse(log, number_of_tasks=6):
        number_of_experiments = len(log['path1'])

        running_total = None

        for experiment in range(number_of_experiments):
            paths = []
            for task in range(1, number_of_tasks+1):
                paths.append(log['path'+str(task)][experiment])

            reuse = Exp2_plotter.reuse_table(paths)

            if running_total is None: running_total = reuse
            else:
                for i in range(len(running_total)):
                    tmp = np.array(reuse[i]) + np.array(running_total[i])
                    running_total[i] = tmp

        for i in range(len(running_total)):
            running_total[i] = (running_total[i]/number_of_experiments).tolist()

        return running_total, sum([j for i in running_total for j in i])

    @staticmethod
    def reuse_table(paths):
        reuse = np.chararray([len(paths), len(paths)])
        reuse[:] = '.'
        reuse = []
        for i in range(len(paths)):
            row = []
            for j in range(i+1, len(paths)):
                #reuse[i][j] = Analytic.path_overlap(paths[i], paths[j])
                row.append(Analytic.path_overlap(paths[i], paths[j]))
            reuse.append(row)
        return reuse

    @staticmethod
    def usefull_training_pr_task(log, number_of_tasks=6):
        used_training_ratio = {}
        for i in range(1, number_of_tasks+1):
            used_training_ratio['task'+str(i)] = []

        for i in range(len(log['path1'])):
            for tasknr in range(1, 7):
                p = log['path' + str(tasknr)][i]
                train_count = log['train' + str(tasknr)][i]
                used_training_ratio['task' + str(tasknr)].append(Exp2_plotter.used_training_ratio(train_count, p))

        return used_training_ratio

    @staticmethod
    def used_training_ratio(training_counter, path):
        total_training = 0
        training_in_path = 0
        for layer, active_modules in zip(training_counter, path):
            for i, module in enumerate(layer):
                total_training += module
                if i in active_modules:
                    training_in_path += module
        print(training_in_path, path)
        return training_in_path/total_training

    def boxplot_used_training_ratio(self, save_file=None, lock=True):
        offsets = [-0.1, -0.05, 0.05, 0.1, -0.15, 0.15]
        def draw_plot(data, offset, edge_color, fill_color):
            pos = np.arange(self.number_of_tasks)+offset
            bp = ax.boxplot(data, positions=pos, widths=0.1, patch_artist=True, manage_xticks=False)
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)
            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

        fig, ax = plt.subplots()
        plt.title('Used training ratio')


        handles = []
        for i, (exp_name, log) in enumerate(self.logs.items()):
            ratio_list = Exp2_plotter.usefull_training_pr_task(log)
            boxes = [None]*self.number_of_tasks
            for k, v in ratio_list.items():
                boxes[int(k[-1])-1] = v

            draw_plot(boxes, offsets[i], self.colors[i], "white")
            handles.append(mpatches.Patch(color=self.colors[i], label=exp_name))

        plt.xticks(range(self.number_of_tasks+1))
        plt.legend(handles=handles)
        plt.xlabel('Task')
        plt.ylabel('Usefull training ratio')

        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)

