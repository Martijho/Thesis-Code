from matplotlib import pyplot as plt
from datetime import datetime
import pickle as pkl
import numpy as np


class Analytic:
    def __init__(self, pathnet):
        self.pathnet = pathnet

    def plot_training_counter(self, lock=True):
        network = np.array(self.pathnet.training_counter).transpose()

        p = plt.imshow(network, cmap='hot', interpolation='nearest', vmin=0)
        plt.colorbar()
        p.axes.get_xaxis().set_visible(False)
        p.axes.get_yaxis().set_visible(False)
        plt.show(block=lock)

    @staticmethod
    def training_along_path(p, training_counter):

        modules_training_log = []
        total = 0
        number_of_modules_in_path = 0
        for layer in range(len(p)):
            l = []
            for module in p[layer]:
                l.append(training_counter[layer][module])
                total += training_counter[layer][module]
                number_of_modules_in_path += 1
            modules_training_log.append(l)
        return modules_training_log, total/number_of_modules_in_path

    '''
    def show_optimal_paths(self):
        for i, t in enumerate(self.pathnet._tasks):
            p = t.optimal_path
            if p is None: continue

            print('='*20, 'Task nr'+str(i+1), '='*20)
            print('Path:')
            print(p)
            print('Training counter:')

            modules_training_log = []
            total = 0
            number_of_modules_in_path = 0
            for layer in range(self.pathnet.depth):
                l = []
                for module in p[layer]:
                    l.append(self.pathnet.training_counter[layer][module])
                    total+=self.pathnet.training_counter[layer][module]
                    number_of_modules_in_path+=1
                modules_training_log.append(l)

            print(modules_training_log)
            print('Average epochs trained on each module:', total/number_of_modules_in_path)
            print('\n')
    '''
    def _is_module_trainable(self, layer, module):
        return self.pathnet._layers[layer].is_module_trainable(module)

    def _module_size(self, layer, module):
        return len(self.pathnet._layers[layer]._modules[module])

    '''
    def show_locked_modules(self):
        pn = self.pathnet
        print('='*20, 'Locked Modules', '='*20)
        for m in range(pn.width):
            print(end='\t')
            for l in range(pn.depth):
                if self._is_module_trainable(l, m):
                    print('-'*self._module_size(l, m), end='   ')
                else:
                    print('X'*self._module_size(l, m), end='   ')
            print()
        print('='*56, end='\n\n')

    def print_training_counter(self):
        pn = self.pathnet
        print('='*19, 'Training counter', '='*19)
        for i in range(self.pathnet.width):
            print(end='\t')
            for j in range(self.pathnet.depth):
                if self.pathnet.training_counter[j][i] == 0:
                    print('-'.ljust(5), end='')
                else:
                    print(str(self.pathnet.training_counter[j][i]).ljust(5), end='')
            print()
        print('='*56, '\n')
    '''


    def _usefull_training_ratio(self, optimal_path, training_counter):
        training_along_path, avg_for_path = self.training_along_path(optimal_path, training_counter)
        number_of_modules = 0
        for layer in optimal_path:
            number_of_modules += len(layer)

        usefull_training = avg_for_path*number_of_modules
        total_training = 0
        for l in training_counter:
            for m in l:
                total_training+=m

        return usefull_training/total_training

    def build_metrics(self, training_counter, optimal_path, log):
        metrics = {}
        metrics['usefull_ratio'] = self._usefull_training_ratio(optimal_path, training_counter)

        metrics['fitness'] = []
        metrics['# modules'] = []
        metrics['avg_training / # modules'] = []
        for p, f, t in zip(log['path'], log['fitness'], log['avg_training']):
            path_a, path_b = p
            fit_a,  fit_b  = f
            avg_a,  avg_b  = t

            metrics['fitness'].append(sum(fit_a)/len(fit_a))
            metrics['fitness'].append(sum(fit_b)/len(fit_b))
            for path in p:
                number_of_modules = 0
                for layer in path:
                    number_of_modules += len(layer)
                metrics['# modules'].append(number_of_modules)

            metrics['avg_training / # modules'] = avg_a / metrics['# modules'][-2]
            metrics['avg_training / # modules'] = avg_b / metrics['# modules'][-1]

        return metrics

    @staticmethod
    def path_overlap(p1, p2):
        counter = 0

        for l1, l2 in zip(p1, p2):
            for m in l1:
                if m in l2:
                    counter += 1

        return counter

    @staticmethod
    def three_path_overlap(p1, p2, p3):
        reuse = [0, 0, 0]
        reuse[0] = Analytic.path_overlap(p1, p2)

        for i in range(len(p1)):

            for m in p3[i]:
                if m in p1[i]:
                    reuse[1]+=1
                elif m in p2[i]:
                    reuse[2]+=1

        return reuse

    @staticmethod
    def encode_path(path, pathnet_width):
        encoded = []
        for l in path:
            layer = np.zeros(pathnet_width)
            layer[l] = 1
            encoded.append(layer)
        return encoded

    @staticmethod
    def pairwise_hamming(population, pathnet_width):
        tmp = []
        for p in population:
            tmp.append(Analytic.encode_path(p, pathnet_width))
        population = tmp

        H = [0, 0, 0]
        for layer in range(3):
            for j in range(len(population) - 1):
                for j_ in range(j + 1, len(population)):
                    for i in range(pathnet_width):
                        H[layer] += abs(population[j][layer][i] - population[j_][layer][i])

        return [sum(H)]

    @staticmethod
    def homemade_diversity(population, pathnet_width):
        encoded_population = []
        for path in population: encoded_population.append(Analytic.encode_path(path, pathnet_width))

        average_encoded = []

        for i in range(len(encoded_population[0])):
            running_total = np.zeros(pathnet_width)
            for path in encoded_population:
                running_total+=path[i]
            running_total = running_total / len(population)
            average_encoded.append(running_total)

        avg_dist = []

        for i in range(len(average_encoded)):
            dist = 0
            for ind in encoded_population:
                dist += sum(np.absolute(ind[i] - average_encoded[i]))
            dist /= len(population)
            avg_dist.append(dist)

        return avg_dist

    @staticmethod
    def euclidean_centroid_diversity(population, pathnet_width):
        encoded_population = []
        for path in population: encoded_population.append(Analytic.encode_path(path, pathnet_width))

        centroid = []

        for i in range(len(encoded_population[0])):
            running_total = np.zeros(pathnet_width)
            for path in encoded_population:
                running_total+=path[i]
            running_total = running_total / len(population)
            centroid.append(running_total)

        avg_dist = []

        for i in range(len(centroid)):
            dist = 0
            for ind in encoded_population:
                dist += np.linalg.norm(ind[i]-centroid[i])
            dist /= len(population)
            avg_dist.append(dist)

        return avg_dist

if __name__ == "__main__":
    print('why run this file?')

    pop = [[[1]]]

    for i in pop:
        print(i)

    Analytic.population_diversity(pop, 4)
