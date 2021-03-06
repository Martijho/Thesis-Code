from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import Adagrad, Adam, RMSprop, SGD
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
from datetime import datetime
from layers import Layer, DenseLayer, ConvLayer, TaskContainer
import numpy as np
import pickle
import os

import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PathNet:
    def __init__(self, input_shape=None, width=-1, depth=-1, max_models_before_reset=30):
        K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=4))))
        self._models_created_in_current_session = 0
        self._max_active_modules = max_models_before_reset

        self._tasks = []

        self.training_counter = [[0]*width for _ in range(depth)]
        self.depth = depth
        self.width = width
        self.input_shape = input_shape
        self._layers = []
        self.max_modules_pr_layer = 2
        self.min_modules_pr_layer = 1

    @staticmethod
    def binary_mnist():
        config                  = [{'out': 20, 'activation': 'relu'}]
        input_shape             = [28, 28, 1]
        output_size             = 2
        depth                   = 3
        width                   = 10
        max_modules_pr_layer    = 3
        learning_rate           = 0.0001
        optimizer_type          = SGD
        loss                    = 'binary_crossentropy'

        layers = []
        for l in range(depth):
            if len(layers) == 0:
                layers.append(DenseLayer(width, 'L0', config, flatten=True))
            else:
                layers.append(DenseLayer(width, 'L'+str(l), config))


        Layer.initialize_whole_network(layers, input_shape)

        task = TaskContainer(input_shape, output_size, name='unique_binary_mnist',
                             optimizer=optimizer_type, loss=loss, lr=learning_rate)

        pathnet = PathNet(input_shape=input_shape, width=width, depth=depth)
        pathnet._layers = layers
        pathnet._tasks = [task]
        pathnet.max_modules_pr_layer = max_modules_pr_layer

        for layer in pathnet._layers:
            layer.save_initialized_weights()

        return pathnet, task

    @staticmethod
    def mnist(output_size=10, image_shape=None):
        conv_config             = [{'channels': 1, 'kernel': (3, 3), 'stride': (1, 1), 'activation': 'relu'}]
        config                  = [{'out': 20, 'activation': 'relu'}]
        input_shape             = [28, 28, 1] if image_shape is None else image_shape
        output_size             = output_size
        depth                   = 3
        width                   = 10
        max_modules_pr_layer    = 3
        min_modules_pr_layer    = 1
        learning_rate           = 0.0001
        optimizer_type          = Adam
        loss                    = 'categorical_crossentropy'
        flatten_in_unique       = True

        layers = []

        layers.append(ConvLayer(width, 'L0', conv_config))
        layers.append(ConvLayer(width, 'L1', conv_config))
        layers.append(ConvLayer(width, 'L2', conv_config, maxpool=True))

        print(input_shape)

        Layer.initialize_whole_network(layers, input_shape)

        task = TaskContainer(input_shape, output_size, flatten_in_unique, name='unique_mnist',
                             optimizer=optimizer_type, loss=loss, lr=learning_rate)

        pathnet = PathNet(input_shape=input_shape, width=width, depth=depth, max_models_before_reset=20)
        pathnet._layers = layers
        pathnet._tasks = [task]
        pathnet.max_modules_pr_layer = max_modules_pr_layer
        pathnet.min_modules_pr_layer = min_modules_pr_layer

        for layer in pathnet._layers:
            layer.save_initialized_weights()

        p = pathnet.random_path()
        pathnet.path2model(p, task)

        return pathnet, task

    @staticmethod
    def cifar10():
        conv_config  = [{'channels': 3, 'kernel': (3, 3), 'stride': (1, 1), 'activation': 'relu'}]
        dense_config = [{'out': 20, 'activation': 'relu'}]
        input_shape = [32, 32, 3]
        output_size = 10
        depth = 3
        width = 10
        max_modules_pr_layer = 3
        learning_rate = 0.001
        optimizer_type = Adam
        loss = 'categorical_crossentropy'

        layers = []
        layers.append(ConvLayer(width, 'L0', conv_config))
        layers.append(ConvLayer(width, 'L1', conv_config))
        layers.append(ConvLayer(width, 'L2', conv_config, maxpool=True))
        #layers.append(DenseLayer(width, 'L2', dense_config, flatten=True))

        Layer.initialize_whole_network(layers, input_shape)

        task = TaskContainer(input_shape, output_size, True, name='unique_cifar10',
                             optimizer=optimizer_type, loss=loss, lr=learning_rate)

        pathnet = PathNet(input_shape=input_shape, width=width, depth=depth)
        pathnet._layers = layers
        pathnet._tasks = [task]
        pathnet.max_modules_pr_layer = max_modules_pr_layer

        for layer in pathnet._layers:
            layer.save_initialized_weights()

        return pathnet, task

    @staticmethod
    def overfit_experiment():
        conv_config = [{'channels': 2, 'kernel': (3, 3), 'stride': (1, 1), 'activation': 'relu'}]
        input_shape = [32, 32, 3]
        output_size = 10
        depth = 3
        width = 6
        max_modules_pr_layer = 3
        min_modules_pr_layer = 1
        learning_rate = 0.0001
        optimizer_type = Adam
        loss = 'categorical_crossentropy'
        flatten_in_unique = True

        layers = []
        layers.append(ConvLayer(width, 'L0', conv_config))
        layers.append(ConvLayer(width, 'L1', conv_config))
        layers.append(ConvLayer(width, 'L2', conv_config, maxpool=True))

        Layer.initialize_whole_network(layers, input_shape)

        task = TaskContainer(input_shape, output_size, flatten_in_unique, name='unique_1',
                             optimizer=optimizer_type, loss=loss, lr=learning_rate)

        pathnet = PathNet(input_shape=input_shape, width=width, depth=depth, max_models_before_reset=80)
        pathnet._layers = layers
        pathnet._tasks = [task]
        pathnet.max_modules_pr_layer = max_modules_pr_layer
        pathnet.min_modules_pr_layer = min_modules_pr_layer

        for layer in pathnet._layers:
            layer.save_initialized_weights()

        p = pathnet.random_path()
        pathnet.path2model(p, task)

        return pathnet, task

    @staticmethod
    def search_experiment(output_size=10, image_shape=None):
        conv_config             = [{'channels': 2, 'kernel': (3, 3), 'stride': (1, 1), 'activation': 'relu'}]
        dens_config             = [{'out': 20, 'activation': 'relu'}]
        input_shape             = image_shape if not None else [32, 32, 3]
        output_size             = output_size
        depth                   = 3
        width                   = 20
        max_modules_pr_layer    = 3
        min_modules_pr_layer    = 1
        learning_rate           = 0.0001
        optimizer_type          = Adam
        loss                    = 'categorical_crossentropy'
        flatten_in_unique       = True

        layers = []
        layers.append(ConvLayer(width, 'L0', conv_config))
        layers.append(ConvLayer(width, 'L1', conv_config))
        layers.append(ConvLayer(width, 'L2', conv_config, maxpool=True))

        Layer.initialize_whole_network(layers, input_shape)

        task = TaskContainer(input_shape, output_size, flatten_in_unique, name='unique_1',
                             optimizer=optimizer_type, loss=loss, lr=learning_rate)

        pathnet = PathNet(input_shape=input_shape, width=width, depth=depth, max_models_before_reset=80)
        pathnet._layers = layers
        pathnet._tasks = [task]
        pathnet.max_modules_pr_layer = max_modules_pr_layer
        pathnet.min_modules_pr_layer = min_modules_pr_layer

        for layer in pathnet._layers:
            layer.save_initialized_weights()

        p = pathnet.random_path()
        pathnet.path2model(p, task)

        return pathnet, task


    def create_new_task(self, config=None, like_this=None):
        if like_this is not None:
            new = like_this.create_task_like_this()
        elif config is not None:
            new = TaskContainer(config['input'], config['output'], config['flatten_first'], config['name'],
                                config['optim'], config['lr'], config['loss'])
        else: assert False, 'PathNet:create_new_task(): Both params are None'

        self.path2model(self.random_path(), new)
        self._tasks.append(new)
        return new

    def path2layer_names(self, path):
        names = []
        for active, layer in zip(path, self._layers):
            for nr in active:
                module = layer.get_module(nr)
                for node in module:
                    names.append(node.name)
        return names

    def path2model(self, path, task, stop_session_reset=False):
        self._models_created_in_current_session += 1
        if not stop_session_reset and self._models_created_in_current_session >= self._max_active_modules:
            self.reset_backend_session()

        inp = Input(task.input_shape)
        thread = inp
        for layer, active_modules in enumerate(path):
            thread = self._layers[layer].add_layer_selection(active_modules, thread)

        output = task.add_unique_layer(thread)

        model = Model(inputs=inp, outputs=output)
        if task.optimizer == Adagrad or task.optimizer == Adam or task.optimizer == RMSprop:
            optim = task.optimizer()
        else:
            optim = task.optimizer(task.learningrate)
        model.compile(optimizer=optim, loss=task.loss, metrics=['accuracy'])

        return model

    def save_new_optimal_path(self, path, task):
        if task.optimal_path is not None: return

        if task not in self._tasks:
            self._tasks.append(task)

        task.optimal_path = path
        for l, layer in enumerate(path):
            self._layers[l].lock_modules(layer)

        for layer in self._layers:
            layer.reinitialize_if_open()

        new_counter = np.zeros_like(self.training_counter)
        for stored_task in self._tasks:
            if stored_task.optimal_path is None: continue
            p = stored_task.optimal_path
            for layer in range(self.depth):
                for module in p[layer]:
                    new_counter[layer][module] = self.training_counter[layer][module]
        self.training_counter = new_counter

    def random_path(self, min=1, max=2):
        if min < 1:
            min = 1

        max = self.max_modules_pr_layer
        min = self.min_modules_pr_layer
        path = []
        for _ in range(self.depth):
            contenders = list(range(self.width))
            np.random.shuffle(contenders)
            if min == max:
                path.append(contenders[:min])
            else:
                path.append(contenders[:np.random.randint(low=min, high=max+1)])

        return path
        
    def increment_training_counter(self, path):
        for layer in range(self.depth):
            for module in path[layer]:
                if self._layers[layer].is_module_trainable(module):
                    self.training_counter[layer][module] += 1

    def train_path(self, x, y, path=None, epochs=5, batch_size=64, verbose=True, model=None, validation_split=0.2):
        if path is None:
            path = self.random_path(max=3)
        if model is None:
            model = self.path2model(path)


        hist = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)


        for _ in range(epochs):
            self.increment_training_counter(path)

        return model, path, hist.history

    def evaluate_path(self, x, y, path=None, task=None, model=None):
        if model is None:
            model = self.path2model(path, task)
        predictions = model.predict(x)

        hit = 0
        miss = 0
        for p, t in zip(predictions, y):
            if np.argmax(p) == np.argmax(t):
                hit+=1
            else:
                miss+=1

        return hit/(hit+miss)

    def reset_backend_session(self):
        for layer in self._layers:
            layer.save_layer_weights()

        for task in self._tasks:
            task.save_layer_weights()

        K.clear_session()

        for layer in self._layers:
            layer._init_layer()

        Layer.initialize_whole_network(self._layers, self.input_shape)

        for layer in self._layers:
            layer.load_layer_weights()

        for task in self._tasks:
            task.load_layer_weights()

        self._models_created_in_current_session = 1 +len(self._tasks)

    def save_pathnet(self):
        layer_logs = []
        for layer in self._layers:
            layer_logs.append(layer.get_layer_log())

        task_logs = []
        for task in self._tasks:
            task_logs.append(task.get_task_log())

        log = {
            'depth': self.depth,
            'width': self.width,
            'in_shape': self.input_shape,
            'training_counter': self.training_counter,
            'max_modules_pr_layer': self.max_modules_pr_layer,
            'min_modules_pr_layer': self.min_modules_pr_layer,
            'layer_logs': layer_logs,
            'task_logs': task_logs
            }

        now = datetime.now()
        with open('../logs/pathnet_structure.pkl', 'wb') as f:
            pickle.dump(log, f)

    @staticmethod
    def load_pathnet(filename):
        log = None
        with open(filename, 'rb') as f:
            log = pickle.load(f)

        layers = []
        for layer_log in log['layer_logs']:
            if layer_log['layer_type'] == 'dense':
                layers.append(DenseLayer.build_from_log(layer_log))
            if layer_log['layer_type'] == 'conv':
                layers.append(ConvLayer.build_from_log(layer_log))

        Layer.initialize_whole_network(layers, log['in_shape'])
        for layer, layer_log in zip(layers, log['layer_logs']):
            layer.load_layer_log(layer_log)

        pathnet = PathNet(input_shape=log['in_shape'], width=log['width'], depth=log['depth'])
        pathnet._layers = layers
        pathnet.training_counter = log['training_counter']
        pathnet.max_modules_pr_layer = log['max_modules_pr_layer']
        pathnet.min_modules_pr_layer = log['min_modules_pr_layer']

        tasks = []
        for task_log in log['task_logs']:
            task = TaskContainer.build_from_log(task_log)
            pathnet.path2model(pathnet.random_path(), task)
            task.layer.set_weights(task_log['layer_weights'])
            tasks.append(task)

        pathnet._tasks = tasks

        return pathnet
