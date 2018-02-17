from __future__ import print_function
import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('../')
from path_search import TS_box
from analytic import Analytic
import time as t
generations = 500

def max_proportion(pop):
    total_max = np.sum(pop == np.max(pop))
    return total_max/len(pop)

def foo(S, N):
    population = np.random.normal(0.5, 0.2, N)
    ratio = [max_proportion(population)]

    for i in range(generations):
        selection, ind = TS_box.selection(population, S)

        if S == 3:
            selection, ind = zip(*list(reversed(sorted(list(zip(selection, ind))))))
            population[ind[2]] = (selection[0]+selection[1])/2
        else:
            population[ind] = np.max(selection)

        ratio.append(max_proportion(population))

    return ratio


def plot_simulated_convergance_time():

    runs = 100
    pressures = [3, 2, 5, 10, 15, 20, 25]
    N = 64

    for s in pressures:
        print('pressure', s)
        results = []

        for i in range(runs):
            results.append(foo(s, N))

        averages = []

        for i in range(generations):
            total = 0
            for r in results:
                total+=r[i]
            total /= len(results)

            averages.append(total)

        plt.plot(averages)

    plt.legend(['3+recomb', '2', '5', '10', '15', '20', '25'])
    plt.show()

def calculate_population_diversity_by_layer_in_random_initialization():
    t = np.zeros(3)
    for i in range(1000):
        pop = []
        for _ in range(64):
            max = 3
            min = 1
            path = []
            for _ in range(3):
                contenders = list(range(20))
                np.random.shuffle(contenders)
                if min == max:
                    path.append(contenders[:min])
                else:
                    path.append(contenders[:np.random.randint(low=min, high=max + 1)])
            pop.append(path)


        similarity_metric = Analytic.population_diversity(pop, 20)
        t += np.array(similarity_metric)

    print(t/1000)


