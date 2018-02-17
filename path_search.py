from __future__ import print_function
from __future__ import print_function
from analytic import Analytic
from reprint import output
import numpy as np
import random
import time
import copy


class PathSearch:
    def __init__(self, pathnet):
        self.pathnet = pathnet

    def tournamet_search(self, x, y, task=None, stop_when_reached=0.99, hyperparam=None):
        if hyperparam is None:
            batch_size          = 16
            training_iterations = 50
            population_size     = 64
        else:
            batch_size          = hyperparam['batch_size']
            training_iterations = hyperparam['training_iterations']
            population_size     = hyperparam['population_size']

        if 'generation_limit' in hyperparam.keys():
            generation_limit = hyperparam['generation_limit']
        else:
            generation_limit = 500

        population = [self.pathnet.random_path() for _ in range(population_size)]
        fitness = [' ']*population_size
        generation = 0
        t_start = 0
        t_stop = 0

        log = {'path':[], 'fitness':[], 'avg_training':[]}

        if task is None: task = self.pathnet._tasks[-1]

        with output(output_type='list') as output_lines:
            output_lines[0] = '='*15+' Generation 0 ' + '='*15
            for ind in population:
                output_lines.append(str(ind).ljust(55)+'-'*4)

            while generation < generation_limit:

                generation += 1


                output_lines[0] = output_lines[0] = '='*15+' Generation '+str(generation)+' ' + '='*15 + '\t' + str(t_stop-t_start)
                t_start = time.time()

                ind = random.sample(range(population_size), 2)
                a, b = ind[0], ind[1]

                path_a = population[a]
                path_b = population[b]


                if generation == 1:
                    model_a = self.pathnet.path2model(path_a, task=task, stop_session_reset=True)
                else:
                    model_a = self.pathnet.path2model(path_a, task=task, stop_session_reset=False)

                model_b = self.pathnet.path2model(path_b, task=task, stop_session_reset=True)
                fit_a, fit_b = 0.0, 0.0
                a_hist, b_hist = [], []

                for batch_nr in range(training_iterations):
                    batch = np.random.randint(0, len(x), batch_size)

                    a_hist.append(model_a.train_on_batch(x[batch], y[batch])[1])
                    b_hist.append(model_b.train_on_batch(x[batch], y[batch])[1])
                    fit_a += a_hist[-1]
                    fit_b += b_hist[-1]

                self.pathnet.increment_training_counter(path_a)
                self.pathnet.increment_training_counter(path_b)

                fit_a /= training_iterations
                fit_b /= training_iterations

                log['path'].append([path_a, path_b])
                log['fitness'].append([a_hist, b_hist])
                _, avg_training_a = Analytic.training_along_path(path_a, self.pathnet.training_counter)
                _, avg_training_b = Analytic.training_along_path(path_b, self.pathnet.training_counter)
                log['avg_training'].append([avg_training_a, avg_training_b])


                if fit_a > fit_b:
                    winner, looser = path_a, path_b
                    w_fit = fit_a
                    w_ind, l_ind = a, b
                else:
                    winner, looser = path_b, path_a
                    w_fit = fit_b
                    w_ind, l_ind = b, a


                if w_fit >= stop_when_reached:
                    for _ in range(3): output_lines.append(' ')
                    return winner, w_fit, log

                fitness[w_ind] = w_fit
                fitness[l_ind] = w_fit
                w = [winner]
                GA_box.mutate(w, mutation_probability=1/9, width=self.pathnet.width)
                population[l_ind] = w[0]

                output_lines[w_ind+1] =  str(winner).ljust(55) + ('%.1f' % (w_fit*100))+' %'
                output_lines[l_ind+1] = str(looser).ljust(55) + '\t'*3 + '['+str(w_fit)+']'

                t_stop = time.time()

            max_fit = max(fitness)
            max_path = population[fitness.index(max_fit)]
            for _ in range(3): output_lines.append(' ')
            return max_path, max_fit, log

    def evolutionary_search(self, x, y, task, hyperparam=None, verbose=True):
        if hyperparam is None: hyperparam = {}

        training_iterations = 50 if 'training_iterations' not in hyperparam else hyperparam['training_iterations']
        batch_size          = 16 if 'batch_size'          not in hyperparam else hyperparam['batch_size']
        population_size     = 16 if 'population_size'     not in hyperparam else hyperparam['population_size']
        generation_limit    = 50 if 'generation_limit'    not in hyperparam else hyperparam['generation_limit']
        threshold_acc       = 0.95 if 'threshold_acc'     not in hyperparam else hyperparam['threshold_acc']
        mutation_prob       = 1/(self.pathnet.max_modules_pr_layer * self.pathnet.depth)

        if verbose:
            print('\n\t'*3, 'Evolutionary Search\n\tHyper parameters:')
            if 'name' in hyperparam: print('\t\t\t', hyperparam['name'])
            print('\t\tTraining_iterations: ', training_iterations)
            print('\t\tbatch_size:          ', batch_size)
            print('\t\tpopulation_size:     ', population_size)
            print('\t\tgeneration_limit:    ', generation_limit)
            print('\t\tthreshold_acc:       ', threshold_acc)
            print('\t\tmutation_prob:       ', mutation_prob, '\n\n')


        population = [self.pathnet.random_path() for _ in range(population_size)]
        population_history = []
        fitness_history    = []

        with output(output_type='list') as output_lines:
            if verbose:
                output_lines[0] = '='*15+' Generation 0 ' + '='*15
                for ind in population:
                    output_lines.append(str(ind).ljust(55)+'-'*4)
                for _ in range(2): output_lines.append(' ')


            for generation in range(1, generation_limit+1):
                if verbose:
                    output_lines[0] = output_lines[0] = '='*15 + ' Generation ' + str(generation) + ' ' + '='*15

                if verbose:
                    fitness = GA_box.train_then_eval(self.pathnet, population, x, y, task, training_iterations=50,
                                                         batch_size=16, output_lines=output_lines, evaluations=50*16)
                else:
                    fitness = GA_box.train_then_eval(self.pathnet, population, x, y, task, training_iterations=50,
                                                     batch_size=16, evaluations=50 * 16)
                population_history.append(population)
                fitness_history.append(fitness)

                population, fitness = GA_box.sort_by_fitness(population, fitness)
                population, fitness = GA_box.selection(population, fitness)

                if fitness[0] >= threshold_acc: break

                children = GA_box.recombination(population)

                GA_box.mutate(children, mutation_probability=mutation_prob, width=self.pathnet.width)

                if verbose:
                    i = 1
                    for ind, fit in zip(population, fitness):
                        output_lines[i] = str(ind).ljust(55)+('%.1f' % (fit*100)) + ' %' + ' '*20
                        i+=1
                    for ind in children:
                        output_lines[i] = str(ind).ljust(55) + '-'*4 + ' '*20
                        i+=1

                population += children

            if verbose:
                for i, (ind, fit) in enumerate(zip(population, fitness)):
                    output_lines[i+1] = str(ind).ljust(55) + ('%.1f' % (fit * 100)) + ' %'+ ' '*20

        log = {'population': population_history,
               'fithess': fitness_history}
        return population[0], fitness[0], log

    def new_tournamet_search(self, x, y, task=None, hyperparam=None, verbose=True):
        if hyperparam is None: hyperparam = {}

        training_iterations = 50   if 'training_iterations' not in hyperparam else hyperparam['training_iterations']
        batch_size          = 16   if 'batch_size'          not in hyperparam else hyperparam['batch_size']
        population_size     = 16   if 'population_size'     not in hyperparam else hyperparam['population_size']
        generation_limit    = 500  if 'generation_limit'    not in hyperparam else hyperparam['generation_limit']
        threshold_acc       = 0.95 if 'threshold_acc'       not in hyperparam else hyperparam['threshold_acc']
        selection_pressure  = 2    if 'selection_pressure'  not in hyperparam else hyperparam['selection_pressure']
        replace_func        = TS_box.replace_2to2 if 'replace_func' not in hyperparam else hyperparam['replace_func']
        mutation_prob       = 1.0/(self.pathnet.max_modules_pr_layer * self.pathnet.depth)
        if replace_func is None: replace_func = TS_box.replace_2to2

        if verbose:
            print('\n\t'*3, 'Tournament Search\n\tHyper parameters:')
            if 'name' in hyperparam: print('\t\t\t', hyperparam['name'])
            print('\t\tTraining_iterations: ', training_iterations)
            print('\t\tbatch_size:          ', batch_size)
            print('\t\tpopulation_size:     ', population_size)
            print('\t\tgeneration_limit:    ', generation_limit)
            print('\t\tthreshold_acc:       ', threshold_acc)
            print('\t\tselection_pressure: ', selection_pressure)
            print('\t\treplace_func:        ', replace_func)
            print('\t\tmutation_prob:       ', mutation_prob, '\n\n')


        population = [self.pathnet.random_path() for _ in range(population_size)]
        generation = 0
        t_start = 0
        t_stop = 0
        competing = None
        fitness = None
        log = {'paths':[], 'fitness':[], 'training_counter':[], 'selected_paths':[]}

        if task is None: task = self.pathnet._tasks[-1]

        with output(output_type='list') as output_lines:
            if verbose:
                output_lines[0] = '='*15+' Generation 0 ' + '='*15
                for ind in population:
                    output_lines.append(str(ind).ljust(55)+'-'*4)
                for _ in range(3): output_lines.append(' ')

            while generation < generation_limit:

                generation += 1


                output_lines[0] = output_lines[0] = '='*15+' Generation '+str(generation)+' ' + '='*15 + '\t' + str(t_stop-t_start)
                t_start = time.time()

                competing, indecies = TS_box.selection(population, selection_pressure)
                out_channel = output_lines if verbose else None

                training_fitness = TS_box.train(self.pathnet, competing, indecies, x, y,
                                                task, training_iterations=50, batch_size=16, output_lines=out_channel)

                if selection_pressure >= 0:
                    fitness = TS_box.evaluate(self.pathnet, competing, indecies, x, y,
                                              task, evaluation_size=800, output_lines=out_channel)
                else: fitness = training_fitness

                competing, fitness, indecies = TS_box.sort_by_fitness(competing, fitness, indecies=indecies)

                #Logging
                log['paths'].append(copy.deepcopy(population))
                log['fitness'].append(fitness)
                log['training_counter'].append(copy.deepcopy(self.pathnet.training_counter))
                log['selected_paths'].append(indecies)

                population = replace_func(competing, indecies, population, mutation_probability=mutation_prob,
                                          width=self.pathnet.width)

                if replace_func == TS_box.replace_2to2:         subsection = int(len(indecies)/2)
                if replace_func == TS_box.replace_3to2:         subsection = 2*int(len(indecies)/3)
                if replace_func == TS_box.winner_replace_all:   subsection = 1
                if verbose:
                    for i, pop, fit in zip(indecies[subsection:], competing[subsection:], fitness[subsection:]):
                        output_lines[i+1] = str(population[i]).ljust(54) + ('[%.1f' % (fit*100)) + ' %]'


                if fitness[0] >= threshold_acc or generation == generation_limit: break

                t_stop = time.time()

            final_fitness = TS_box.evaluate(self.pathnet, population, list(range(len(population))), x, y,
                                            task, evaluation_size=800, output_lines=output_lines)
            population, final_fitness, indecies = TS_box.sort_by_fitness(competing, final_fitness,
                                                                         indecies=list(range(len(population))))
            return population[0], final_fitness[0], log


class TS_box:
    @staticmethod
    def selection(population, preassure):
        ind = random.sample(range(len(population)), preassure)
        paths = np.array(population)[ind]
        return paths.tolist(), ind

    @staticmethod
    def train(pathnet, selection, indecies, x, y, task, training_iterations=50, batch_size=16, output_lines=None):
        training_fitness = []

        for i, genome in zip(indecies, selection):
            model = pathnet.path2model(genome, task)
            batch = np.random.randint(0, len(x), batch_size*training_iterations)
            local_fitness = model.fit(x[batch], y[batch], batch_size=16, epochs=1,
                                      verbose=0, validation_split=0.0).history['acc'][0]

            pathnet.increment_training_counter(genome)
            training_fitness.append(local_fitness)

            if output_lines is not None:
                output_lines[i + 1] = str(genome).ljust(54) + ('<%.1f' % (local_fitness * 100)) + ' %>'

        return training_fitness

    @staticmethod
    def evaluate(pathnet, selection, indecies, x, y, task, evaluation_size=800, output_lines=None):
        fitness = []
        for i, genome in zip(indecies, selection):
            model = pathnet.path2model(genome, task)
            batch = np.random.randint(0, len(x), evaluation_size)
            fit = model.evaluate(x[batch], y[batch], batch_size=16, verbose=False)[1]
            fitness.append(fit)
            output_lines[i+1] = output_lines[i+1][:54] + (' %.1f' % (fit * 100)) + ' %'

        return fitness
    '''
    @staticmethod
    def evaluate(pathnet, selection, indecies, x, y, task, training_iterations=50, batch_size=16, output_lines=None):
        pathnet.reset_backend_session()
        fitness = []

        for i, genome in zip(indecies, selection):

            model = pathnet.path2model(genome, task, stop_session_reset=True)
            local_fitness = 0

            for batch_nr in range(training_iterations):
                batch = np.random.randint(0, len(x), batch_size)

                hist = model.train_on_batch(x[batch], y[batch])[1]
                local_fitness += hist

            local_fitness/=training_iterations

            pathnet.increment_training_counter(genome)
            fitness.append(local_fitness)

            if output_lines is not None:
                output_lines[i+1] = str(genome).ljust(55) + ('%.1f' % (local_fitness * 100)) + ' %'

        return fitness
    '''
    @staticmethod
    def sort_by_fitness(selection, fitness, indecies=None):
        if indecies is None:
            return GA_box.sort_by_fitness(selection, fitness)
        else:
            fit, pop, ind = zip(*list(reversed(sorted(list(zip(fitness, selection, indecies))))))
            return list(pop), list(fit), list(ind)

    @staticmethod
    def replace_2to2(competing, indecies, population, mutation_probability=0.1, width=10):
        comp = list(competing)
        ind = list(indecies)
        while len(comp) > 0:

            winner_p = comp.pop(0)
            looser_i = ind.pop(-1)

            ind.pop(0)
            comp.pop(-1)

            child = [winner_p]
            GA_box.mutate(child, mutation_probability=mutation_probability, width=width)
            population[looser_i] = child[0]
        return population
    @staticmethod
    def replace_3to2(competing, indecies, population, mutation_probability=0.1, width=10):
        comp = list(competing)
        ind = list(indecies)
        while len(comp) > 0:

            mom_p = comp.pop(0)
            dad_p = comp.pop(0)
            looser_i = ind.pop(-1)

            ind.pop(0)
            ind.pop(0)
            comp.pop(-1)

            child = GA_box.recombination([mom_p, dad_p])
            GA_box.mutate(child, mutation_probability=mutation_probability, width=width)
            population[looser_i] = child[0]
        return population
    @staticmethod
    def winner_replace_all(competing, indecies, population, mutation_probability=0.1, width=10):
        for looser_ind in indecies[1:]:
            child = [competing[0]]
            GA_box.mutate(child, mutation_probability=mutation_probability, width=width)
            population[looser_ind] = child[0]
        return population
    @staticmethod
    def recombination(mom, dad):
        child = []

        for i, (mom, dad) in enumerate(zip(mom, dad)):
            if i % 2 == 0:
                child.append(mom)
            else:
                child.append(dad)

        return child


class GA_box:

    @staticmethod
    def evaluate_population(pathnet, population, x, y, task, training_iterations=50, batch_size=16,
                            output_lines=None):
        pathnet.reset_backend_session()
        fitness = []

        for i, genome in enumerate(population):

            model = pathnet.path2model(genome, task, stop_session_reset=True)
            local_fitness = 0
            for batch_nr in range(training_iterations):
                batch = np.random.randint(0, len(x), batch_size)

                hist = model.train_on_batch(x[batch], y[batch])[1]
                local_fitness += hist

            local_fitness/=training_iterations

            pathnet.increment_training_counter(genome)
            fitness.append(local_fitness)

            if output_lines is not None:
                output_lines[i+1] = str(genome).ljust(54) + ('[%.1f' % (local_fitness * 100)) + ' %]'

        return fitness

    @staticmethod
    def train_then_eval(pathnet, population, x, y, task, training_iterations=55, batch_size=16,
                            output_lines=None, evaluations=55):
        GA_box.evaluate_population(pathnet, population, x, y, task, training_iterations=training_iterations,
                                   batch_size=batch_size, output_lines=output_lines)
        pathnet.reset_backend_session()
        fitness = []
        for i, genome in enumerate(population):
            batch = np.random.randint(0, len(x), evaluations)
            fit = pathnet.evaluate_path(x[batch], y[batch], genome, task)
            fitness.append(fit)
            output_lines[i+1] += 'Eval['+str(evaluations)+'] = ' + ('%.1f' % (fit * 100)) + ' %'

        return fitness
    @staticmethod
    def sort_by_fitness(population, fitness):
        fitness, population = zip(*list(reversed(sorted(list(zip(fitness, population))))))
        return list(population), list(fitness)

    @staticmethod
    def selection(population, fitness):
        return population[:int(len(population)/2)], fitness[:int(len(fitness)/2)]

    @staticmethod
    def recombination(population):
        children = []

        for _ in population:
            parents = random.sample(range(len(population)), 2)
            child = []

            for i, (mom, dad) in enumerate(zip(population[parents[0]], population[parents[1]])):
                if i % 2 == 0: child.append(mom)
                else:          child.append(dad)

            children.append(child)

        return children

    @staticmethod
    def mutate(population, mutation_probability=0.1, width=10):
        for i in range(len(population)):
            for j in range(len(population[i])):

                genome = np.array(population[i][j])
                prob   = np.random.uniform(0, 1, size=genome.shape)
                prob   = (prob < mutation_probability) * 1
                shift  = np.random.randint(low=-2, high=3, size=genome.shape)

                shift  *= prob
                genome = ((genome + shift) % width).tolist()

                for m in population[i][j]:
                    if len(set(genome)) == len(population[i][j]): break
                    if m not in genome: genome.append(m)

                population[i][j] = list(set(genome))




'''
    def evaluate(self, population, x, y, task, epochs, batch_size):
        fitness = []
        history = []
        for nr, path in enumerate(population):
            t = time.time()
            if nr == 0:
                model = self.pathnet.path2model(path, task)
            else:
                model = self.pathnet.path2model(path, task, stop_session_reset=True)

            hist = model.fit(x, y, epochs=epochs, validation_split=0.2, verbose=True, batch_size=batch_size)

            self.time_log.append(time.time()-t / model.count_params())

            self.pathnet.increment_training_counter(path)
            history.append(hist.history)
            fitness.append(hist.history['val_acc'][0])

        return fitness, history

    def select_one_index(self, fitness):
        value = random.random() * sum(fitness)
        for i in range(len(fitness)):
            value -= fitness[i]
            if value <= 0:
                return i

    def selection(self, population, fitness):
        population_size = len(population)
        fit = []
        pop = []
        for f, i in sorted(zip(fitness, population)):
            fit.append(f)
            pop.append(i)

        survived = []
        sur_fit = []

        survived.append(pop.pop())
        sur_fit.append(fit.pop())
        del pop[0]
        del fit[0]

        for _ in range(int(len(population) / 2)-1):
            i = self.select_one_index(fit)
            len_1 = len(pop)
            survived.append(pop[i])
            sur_fit.append(fit[i])

            del pop[i]
            del fit[i]
            len_2 = len(pop)

            assert len_1-1 == len_2, 'Selection(EA): removes too much from population-list. Remove "del pop[i]'

        assert len(survived) == population_size/2, 'Selection(EA): wrong number of survived genotypes'

        return survived, sur_fit

    def combine(self, a, b):
        offspring = []
        for layer_number in range(len(a)):
            layer = []
            for m in a[layer_number]:           # copy duplicate modules
                if m in b[layer_number]:
                    layer.append(m)

            layer_size = (len(a[layer_number]) + len(b[layer_number])) / 2  # Size of layer is mean of parents
            if layer_size - int(layer_size) > 0:                            # if sum is odd, randomly favour one parent
                if random.choice([True, False]):
                    layer_size += 0.5

            layer_size = int(layer_size)

            while len(layer) < layer_size:
                layer.append(random.choice(a[layer_number] + b[layer_number]))
                layer = list(set(layer))

            offspring.append(layer)

        return offspring

    def crossover(self, population):
        old_population_length = len(population)
        new_population = []

        for father in range(0, len(population) - 1, 2):
            mother = father + 1
            new_population.append(self.combine(population[father], population[mother]))

        for father in range(len(population)):
            if len(new_population) == len(population):
                break
            mother = father + int(len(population) / 2)
            if mother >= len(population):
                mother -= len(population)
            new_population.append(self.combine(population[father], population[mother]))

        assert len(new_population) == old_population_length, 'Crossover(EA): wrong number of children' \
                                                             ''
        return new_population

    def mutate(self, population):
        mutated = []
        for p in population:
            N = max([len(x) for x in p])
            L = self.pathnet.depth
            mutated.append(self.mutate_path(p, mutation_prob=1/(N*L)))
        return mutated

    def mutate_path(self, path, mutation_prob=0.1):
        mutated_path = []

        for old_layer in path:
            layer = []
            for old_module in old_layer:
                shift = 0
                if np.random.uniform(0, 1) <= mutation_prob:
                    shift = np.random.randint(low=-2, high=3)

                layer.append((old_module + shift) % self.pathnet.width)

            layer = list(set(layer))

            for m in old_layer:
                if len(layer) == len(old_layer):
                    break
                if m not in layer:
                    layer.append(m)

            mutated_path.append(layer)

        return mutated_path

    def sort_generation_by_fitness(self, population, fitness):
        fitness, population = zip(*list(reversed(sorted(list(zip(fitness, population))))))

        population = list(population)
        fitness = list(fitness)

        return population, fitness

    def simple_selection(self, population, fitness):
        return population[:int(len(population)/2)], fitness[:int(len(population)/2)]

    def simple_crossover(self, population):
        new_pop = []
        while len(new_pop) != len(population):
            father = random.choice(population)
            mother = random.choice(population)

            child = []
            for i in range(len(father)):
                layer = []
                if i % 2 == 0:
                    for modules in mother[i]:
                        layer.append(modules)
                else:
                    for modules in father[i]:
                        layer.append(modules)
                child.append(layer)
            new_pop.append(child)

        assert len(new_pop) == len(population), 'Simple_crossover(EA): new population not correct size'
        return new_pop
'''






