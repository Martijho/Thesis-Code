from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from analytic import Analytic
import numpy as np
import time
import matplotlib.lines as mlines
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
        self.colors_by_name = {}
        for k, color in zip(logs.keys(), self.colors): self.colors_by_name[k] = color

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
        return training_in_path/total_training

    @staticmethod
    def search_diversity(population, diversity_metric_function):
        diversity = []
        for i, gen in enumerate(population):
            diversity_metric = diversity_metric_function(gen, 20)
            diversity.append(diversity_metric)

        return diversity

    @staticmethod
    def get_average_search_diversity(log, task_nr, diversity_metric_function):
        diversity = 0

        for search in log['log'+str(task_nr)]:
            search_diversity = []
            for gen in search['paths']:
                div = diversity_metric_function(gen, 20)
                search_diversity.append(sum(div)/len(div))

            diversity = diversity + np.array(search_diversity)

        return diversity / len(log['log1'])



    @staticmethod
    def get_training_fitness_list(log):
        tasks = [[[] for _ in range(100)] for _ in range(6)]
        for met, data in log.items():
            if 'log' in met:
                i = int(met[-1]) - 1
                for example in data:
                    try:
                        for gen, pop in enumerate(example['fitness']):
                            tasks[i][gen].append(sum(pop) / len(pop))
                    except KeyError:
                        for gen, pop in enumerate(example[b'fitness']):
                            tasks[i][gen].append(sum(pop) / len(pop))
        return tasks

    @staticmethod
    def used_capacity(log, L=3, M=20):
        capacity = []
        for i in range(len(log['path1'])):
            used = np.zeros([L, M], dtype=bool)
            for pnr in range(1, 7):
                p = log['path'+str(pnr)][i]
                for layer_nr, layer in enumerate(p):
                    for module in layer:
                        used[layer_nr][module] = True
            capacity.append(np.sum(used))
        return capacity

    @staticmethod
    def average_validation_for_each_search(log):
        acc = []
        for i in range(len(log['eval1'])):
            evaluations = [log['eval'+str(6)][i] for tasknr in range(1, 7)]
            acc.append(sum(evaluations)/len(evaluations))
        return acc

    @staticmethod
    def plot_training_progress_for_one_search_type(log):
        tasks = Exp2_plotter.get_training_fitness_list(log)

        def draw_plot(data, offset, edge_color, fill_color, ax):
            pos = np.arange(100) + offset
            bp = ax.boxplot(data, positions=pos, widths=0.1, patch_artist=True, manage_xticks=False)
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)
            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

        fig, axes = plt.subplots(2, 3)
        plt.title('Training progress')

        for i in range(2):
            for j in range(3):
                draw_plot(tasks[i*3+j], 0, '#dd2020', 'white', axes[i, j])
                #plt.xticks(range(100))
                axes[i, j].set_xlabel('Generation')
                axes[i, j].set_ylabel('Training accuracy')
                axes[i, j].set_title('Task '+str(i*3+j+1))
                axes[i, j].set_ylim([0, 1])
        plt.show()

    @staticmethod
    def capacity_used_during_search(log):

        capacity_list = []
        for exp_nr in range(len(log['log1'])):
            experiment_capacity_list = []
            locked = np.zeros([3, 20], dtype=bool)
            for task_nr in range(1, 7):
                try:
                    generation_list = log['log'+str(task_nr)][exp_nr]['paths']
                except KeyError:
                    generation_list = log['log'+str(task_nr)][exp_nr][b'paths']
                generation_capacity_list = []
                for gen in generation_list:

                    genotype_capacity_list = []
                    for path in gen:
                        genotype = Analytic.encode_path(path, 20)
                        tmp_capacity_used = np.logical_or(locked, genotype)
                        genotype_capacity_list.append(np.sum(tmp_capacity_used))

                    generation_capacity_list.append(genotype_capacity_list)

                try:
                    locked = np.logical_or(locked, Analytic.encode_path(log['path'+str(task_nr)][exp_nr], 20))
                except KeyError:
                    locked = np.logical_or(locked, Analytic.encode_path(log[b'path' + str(task_nr)][exp_nr], 20))

                experiment_capacity_list += generation_capacity_list

            capacity_list.append(experiment_capacity_list)

        return capacity_list

    @staticmethod
    def training_iterations(log):
        training = []
        for i in range(len(log['path1'])):
            mask = np.zeros([3, 20], dtype=bool)
            for task_nr in range(1, 7):
                path = log['path'+str(task_nr)][i]
                mask = np.logical_or(np.array(Analytic.encode_path(path, 20), dtype=bool), mask)
            training.append(np.sum((np.array(log['train6'][1])*mask)))

        return training

    @staticmethod
    def cumulative_validation(log):
        acc = []
        for i in range(len(log['eval1'])):
            total = 0
            for task_nr in range(1, 7):
                total += log['eval'+str(task_nr)][i]
            acc.append(total)

        return acc

    @staticmethod
    def total_capacity(log):
        capacity = []
        for i in range(len(log['path1'])):
            mask = np.zeros([3, 20], dtype=bool)
            for task_nr in range(1, 7):
                path = log['path'+str(task_nr)][i]
                mask = np.logical_or(np.array(Analytic.encode_path(path, 20), dtype=bool), mask)
            capacity.append(np.sum(mask))

        return capacity

    @staticmethod
    def total_reuse(log):
        reuse = 0
        for exp_nr in range(len(log['log1'])):
            experiment_reuse = []
            locked_paths = [log['path1'][exp_nr]]
            for task_nr in range(2, 7):
                try:
                    generation_list = log['log'+str(task_nr)][exp_nr]['paths']
                except KeyError:
                    generation_list = log['log'+str(task_nr)][exp_nr][b'paths']

                for generation in generation_list:
                    generation_reuse = []
                    for genotype in generation:
                        generation_reuse.append(sum([sum(x) for x in Exp2_plotter.reuse_table(locked_paths+[genotype])]))
                    experiment_reuse.append(sum(generation_reuse)/len(generation_reuse))

                locked_paths.append(log['path'+str(task_nr)][exp_nr])
            reuse = reuse + np.array(experiment_reuse)
        reuse = reuse / len(log['log1'])
        return reuse

    @staticmethod
    def get_average_path_size(log, task):
        avg = 0
        for exp in range(len(log['log'+str(task)])):
            experiment_avg = []

            try:             data = log['log'+str(task)][exp]['paths']
            except KeyError: data = log['log'+str(task)][exp][b'paths']

            for generation in data:
                total = 0
                for genotype in generation: total += len(sum(genotype, []))
                experiment_avg.append(total/len(generation))
            avg = avg + np.array(experiment_avg)
        return avg/len(log['log'+str(task)])

    @staticmethod
    def get_average_layer_size(log):

        size_list = []
        for task in range(1, 7):
            avg = 0

            for i, path in enumerate(log['path'+str(task)]):

                sizes = [len(layer) for layer in path]
                avg = avg + np.array(sizes)

            size_list.append(avg/(i+1))
        return size_list

    def plot_training_progress(self, save_file=None, lock=True):
        data = {}
        for k, v in self.logs.items(): data[k] = Exp2_plotter.get_training_fitness_list(v)
        f, axarr = plt.subplots(2, 3, sharex=True, sharey=True)

        handles = [[], [], [], [], [], []]
        for i in range(2):
            for j in range(3):
                ax = axarr[i, j]

                for k, v in data.items():
                    y = np.average(np.array(v[i * 3 + j]), axis=1)

                    '''
                    # Adding scatter to the plot
                    x = np.arange(100)+1
                    for xi in range(6):
                        ax.scatter(x-1, np.array(v[i * 3 + j])[:, i], s=1, color=self.colors_by_name[k], alpha=0.1)
                    '''

                    ax.plot(y, color=self.colors_by_name[k], label=k)
                    handles[i*3+j].append(mpatches.Patch(color=self.colors_by_name[k], label=k))

                ax.set_ylim([0, 1.1])
                ax.set_xlim([-1, 101])
                ax.set_title('Task ' + str(i*3+j+1))
                ax.yaxis.grid(color='gray', linestyle='-', linewidth=2, alpha=0.25)
                if j == 0: ax.set_ylabel('avg Training acc')
                if i == 1: ax.set_xlabel('Generation')
                for k in data.keys():
                    eval_fitness = self.logs[k]['eval' + str(i * 3 + j + 1)]
                    avg_fitness = sum(eval_fitness) / len(eval_fitness)
                    ax.plot([-3, 103], [avg_fitness, avg_fitness], ':', color=self.colors_by_name[k])

        ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
        plt.suptitle('Training Accuracy')

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

    def plot_population_diversity(self, log, name, lock=True, save_file=None, diversity_metric=Analytic.homemade_diversity):
        f, axarr = plt.subplots(2, 3, sharex=True, sharey=True)
        colors = ['red', 'green', 'blue']
        logs = ['log1', 'log2', 'log3', 'log4', 'log5', 'log6']
        for i in range(2):
            for j in range(3):
                ax = axarr[i, j]
                dict_name = logs[i*3+j]

                div_list = []
                for exp_number in range(len(log[dict_name])):
                    try:             pop = log[dict_name][exp_number]['paths']
                    except KeyError: pop = log[dict_name][exp_number][b'paths']
                    div_metric = Exp2_plotter.search_diversity(pop, diversity_metric)
                    div_list.append(div_metric)

                div_vec = np.array(div_list)

                for layer_nr in range(3):
                    top = np.max(div_vec[:, :, layer_nr], axis=0)
                    bot = np.min(div_vec[:, :, layer_nr], axis=0)
                    ax.fill_between(np.arange(100), top, bot, color=colors[layer_nr], alpha=0.2)

                div_average = np.average(div_vec, axis=0)

                ax.set_title('Task ' + str(i*3+j+1))
                ax.plot(div_average[:, 0], label='layer 1', color=colors[0])
                ax.plot(div_average[:, 1], label='layer 2', color=colors[1])
                ax.plot(div_average[:, 2], label='layer 3', color=colors[2])
                if j == 0: ax.set_ylabel('Population Diversity')
                if i == 1: ax.set_xlabel('Generation')
                #ax.set_ylim([0, 5])

        ax.legend(['layer 1', 'layer 2', 'layer 3'], bbox_to_anchor=(1.05, 0),
                  loc='lower left', borderaxespad=0.)
        if diversity_metric == Analytic.homemade_diversity:
            plt.suptitle('Population Diversity(Home-made): '+ name)
        else:
            plt.suptitle('Population Diversity(Pair-wise Hamming): ' + name)

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

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

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)

    def validation_fitness(self, save_file=None, lock=True):
        boxes = {}
        for algo, log in self.logs.items():
            task_fitness = [0, 0, 0, 0, 0, 0]

            print(log['eval1'])
            task_fitness[0] = log['eval1']
            task_fitness[1] = log['eval2']
            task_fitness[2] = log['eval3']
            task_fitness[3] = log['eval4']
            task_fitness[4] = log['eval5']
            task_fitness[5] = log['eval6']

            #task_fitness = (np.array(task_fitness) / len(log['eval1'])).tolist()
            boxes[algo] = task_fitness

        offsets = [-0.1, -0.05, 0.05, 0.1, -0.15, 0.15]
        def draw_plot(data, offset, edge_color, fill_color):
            pos = np.arange(self.number_of_tasks)+offset
            bp = ax.boxplot(data, positions=pos, widths=0.1, patch_artist=True, manage_xticks=False)
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)
            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

        fig, ax = plt.subplots()
        plt.title('Validation accuracy')

        handles = []
        for i, (algo_name, validation) in enumerate(boxes.items()):
            draw_plot(validation, offsets[i], self.colors[i], 'white')
            handles.append(mpatches.Patch(color=self.colors[i], label=algo_name))

        plt.xticks(range(self.number_of_tasks+1))
        plt.legend(handles=handles)
        plt.xlabel('Task')
        plt.ylabel('Validation accuracy')

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

    def plot_capacity_vs_fitness(self, save_file=None, lock=True):
        plt.figure('Capacity effectivity')
        handles = []
        for algo_name, log in self.logs.items():
            x = Exp2_plotter.average_validation_for_each_search(log)
            y = Exp2_plotter.used_capacity(log)

            plt.scatter(x, y, color=self.colors_by_name[algo_name])
            handles.append(mpatches.Patch(color=self.colors_by_name[algo_name], label=algo_name))
        plt.title('PathNet usage')
        plt.xlabel('Average validation fitness')
        plt.ylabel('Used capacity')
        plt.legend(handles=handles)

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)

    def plot_capacity_usage(self, save_file=None, lock=True):
        plt.figure('Used capacity')
        plt.title('Used capacity')

        handles = []
        for i, (exp_name, log) in enumerate(self.logs.items()):
            boxes = np.array(Exp2_plotter.capacity_used_during_search(log))
            boxes = np.average(np.average(boxes, axis=0), axis=1)
            for task_nr in range(6):
                plt.plot(np.arange(100)+(task_nr*100),
                         boxes[task_nr*100:(task_nr+1)*100],
                         color=self.colors_by_name[exp_name])

            handles.append(mpatches.Patch(color=self.colors_by_name[exp_name], label=exp_name))

        for i, line in enumerate([6.00, 11.40, 16.26, 20.64, 24.58, 28.12]):
            x = [i*100, (i+1)*100]
            y = [line, line]
            plt.plot(x, y, ':', color='grey', alpha=0.5)
        handles.append(mpatches.Patch(color='grey', linestyle=':', alpha=0.5, label='Random selection'))

        plt.legend(handles=handles)
        plt.xlabel('Generation')
        plt.ylabel('Capcity')

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

    def plot_training_iterations(self, save_file=None, lock=True):
        min_capacity = 25
        max_capacity = 36

        plt.figure('Training Iterations')
        plt.title('Training Value')
        handles = []
        for algo_name, log in self.logs.items():
            y = Exp2_plotter.training_iterations(log)
            x = Exp2_plotter.cumulative_validation(log)
            s = Exp2_plotter.total_capacity(log)
            s = (np.array(s)-min_capacity)/max_capacity
            s = (s*680) + 20

            plt.scatter(x, y, s=s, color=self.colors_by_name[algo_name], alpha=0.5)
            handles.append(mpatches.Patch(color=self.colors_by_name[algo_name], label=algo_name))

        plt.xlabel('Cumulative Validation Accuracy for all Tasks')
        plt.ylabel('Total Training Iterations')
        plt.legend(handles=handles)

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

    def plot_reuse(self, save_file=None, lock=True):
        plt.figure('Reuse')
        plt.title('Reuse')

        handles = []
        for i, (exp_name, log) in enumerate(self.logs.items()):
            boxes = np.array(Exp2_plotter.total_reuse(log))
            for task_nr in range(1, 6):
                plt.plot(np.arange(100) + task_nr*100,
                         boxes[(task_nr-1)*100:(task_nr)*100],
                         color=self.colors_by_name[exp_name])

            handles.append(mpatches.Patch(color=self.colors_by_name[exp_name], label=exp_name))

        for i, line in enumerate([0.6, 1.14, 1.62, 2.06, 2.46]):
            x = [(i+1)*100, (i+2)*100]
            y = [line, line]
            plt.plot(x, y, ':', color='grey', alpha=0.5)
        handles.append(mpatches.Patch(color='grey', linestyle=':', alpha=0.5, label='Random selection'))

        plt.xticks(np.arange(7)*100)
        plt.legend(handles=handles)
        plt.xlabel('Generation')
        plt.ylabel('Reuse')
        plt.grid(linestyle=':', alpha=0.4)

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

    def plot_average_path_size(self, save_file=None, lock=True):
        f, axarr = plt.subplots(2, 3, sharex=True, sharey=True)

        for i in range(2):
            for j in range(3):
                ax = axarr[i, j]
                task_nr = i*3 + j + 1

                for k, v in self.logs.items():
                    y = Exp2_plotter.get_average_path_size(v, task_nr)

                    ax.plot(y, label=k, color=self.colors_by_name[k])

                ax.set_title('Task ' + str(task_nr))
                ax.grid(linestyle=':', alpha=0.7)
                if j == 0: ax.set_ylabel('Average Modules in paths')
                if i == 1: ax.set_xlabel('Generation')



        ax.legend(list(self.logs.keys()), bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)

        plt.suptitle('Average path size')

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

    def plot_layer_sizes(self, save_file=None, lock=True):
        avg_sizes = {}
        for k, v in self.logs.items(): avg_sizes[k] = Exp2_plotter.get_average_layer_size(v)

        f, axarr = plt.subplots(2, 3, sharex=True, sharey=True)

        exp_positions = [1, 2, 3, 4, 5]
        offsets = [-0.2, 0, 0.2]


        for i in range(2):
            for j in range(3):
                ax = axarr[i, j]
                task_nr = i * 3 + j + 1

                for pos, (k, v) in zip(exp_positions, self.logs.items()):
                    y = avg_sizes[k][i*3+j]
                    ax.bar(pos+np.array(offsets), y, 0.15, label=k, color=self.colors_by_name[k])

                ax.set_xticks(range(1, 6))
                ax.set_xticklabels(list(avg_sizes.keys()))
                ax.set_title('Task ' + str(task_nr))

                ax.plot([0.5, 5.5], [2, 2], ':', color='grey', alpha=0.5)

                if j == 0: ax.set_ylabel('Average Modules in layer')


        plt.suptitle('Average layer size')

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

    def plot_average_population_diversity(self, save_file=None, lock=True, diversity_metric=Analytic.homemade_diversity):
        f, axarr = plt.subplots(2, 3, sharex=True, sharey=True)

        for i in range(2):
            for j in range(3):
                ax = axarr[i, j]

                for exp_name, log in self.logs.items():
                    div = Exp2_plotter.get_average_search_diversity(log, i*3+j+1, diversity_metric)

                    ax.plot(div, label=exp_name, color=self.colors_by_name[exp_name])

                if j == 0: ax.set_ylabel('Population Diversity')
                if i == 1: ax.set_xlabel('Generation')

        ax.legend(list(self.logs.keys()), bbox_to_anchor=(1.05, 0),loc='lower left', borderaxespad=0.)

        plt.suptitle('Population Diversity('+diversity_metric.__name__+')')

        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)

# Overfit experiments
class Exp3_plotter:
    def __init__(self, logs):
        self.logs = logs
        self.colors = ['#dd2020', '#e8bd00', '#7b6500', '#0fcf71', '#409edb', '#e55d82']

    def plot_validation_accuracy(self, save_file=None, lock=True):

        plt.figure('Validation accuracy')

        denovo_pn_x, denovo_pn_y = [], []
        pn_x, pn_y = [], []
        static_x, static_y = [], []
        transfer_x, transfer_y = [], []

        for k, v in self.logs.items():
            for log in v:
                plt.scatter(log['set_size'], log['pathnet_denovo_eval'], alpha=0.4, color=self.colors[0])
                plt.scatter(log['set_size'], log['pathnet_eval'],  alpha=0.4, color=self.colors[1])
                plt.scatter(log['set_size'], log['static_evaluation_fitness'], alpha=0.4, color=self.colors[2])
                plt.scatter(log['set_size'], log['static_transfer_evaluation_fitness_cSVHN'], alpha=0.4, color=self.colors[3])

                plt.scatter(log['set_size'], log['pathnet_denovo_fitness'], marker='x', color=self.colors[0])
                plt.scatter(log['set_size'], log['pathnet_fitness'],  marker='x', color=self.colors[1])

                y = log['static_training_fitness'][-5:]
                #plt.scatter(log['set_size'], sum(y)/len(y), marker='x', color=self.colors[2])

                y = log['static_transfer_training_fitness_cSVHN'][-5:]
                print(log['static_transfer_training_fitness_cSVHN'])
                #plt.scatter(log['set_size'], sum(y)/len(y), marker='x', color=self.colors[3])

        denovo =    mpatches.Patch(color=self.colors[0],  label='PathNet de novo')
        pn =        mpatches.Patch(color=self.colors[1],  label='Pretrained PathNet')
        static =    mpatches.Patch(color=self.colors[2],  label='Static ML model')
        transfer =  mpatches.Patch(color=self.colors[3],  label='Pretrained ML with fine tuning')

        # Create a legend for the first line.
        #first_legend = plt.legend(handles=[denovo, pn, static, transfer], loc=4)
        #ax = plt.gca().add_artist(first_legend)

        vali = plt.scatter([], [], color='grey', marker='x', label='Validation')
        trai = plt.scatter([], [], color='grey', marker='o', alpha=0.4, label='Training')
        #plt.legend(handles=[vali, trai], loc=3)
        plt.legend(handles=[denovo, pn, static, transfer, vali, trai])
        plt.title('Validation accuracy')
        plt.xlabel('Training data size')
        plt.ylabel('Validation accuracy')


        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)

        if save_file is not None:
            plt.savefig(save_file + '.png')
        plt.show(block=lock)
