from dataprep import DataPrep
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.merge import add
from keras.optimizers import Adagrad, Adam, RMSprop, SGD
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

CHANNELS1 = 2
KERNEL1   = [3, 3]
STRIDE1   = [1, 1]

CHANNELS2 = 1
KERNEL2   = [3, 3]
STRIDE2   = [1, 1]

CHANNELS3 = 2
KERNEL3   = [3, 3]
STRIDE3   = [1, 1]


modules = 3
optim = Adam()
loss = 'categorical_crossentropy'
training_iterations = 1000
def conv_module(inp):
    x = Conv2D(CHANNELS1, KERNEL1, strides=STRIDE1, activation='relu')(inp)
    #x = Conv2D(CHANNELS2, KERNEL2, strides=STRIDE2, activation='relu')(x)
    #x = Conv2D(CHANNELS3, KERNEL3, strides=STRIDE3, activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dense(20, activation='relu')(inp)

    return x


def add_layer(inp):
    layer = []
    for _ in range(modules):
        layer.append(conv_module(inp))

    if len(layer) == 1: return layer[0]
    else              : return add(layer)

data = DataPrep()
data.cSVHN_ez()



inp = Input(shape=[32, 32, 3])
x = add_layer(inp)
x = add_layer(x)
x = add_layer(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)

m = Model(input=[inp], output=[x])
m.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
m.summary()

N = m.count_params()*20
val_N = m.count_params()*5

plt.axis([0, training_iterations, 0, 1])
plt.ion()
val_hist = []
tra_hist = []
for i in range(training_iterations):
    local_fitness = 0
    for batch_nr in range(50):
        batch = np.random.randint(0, len(data.x), 16)

        hist = m.train_on_batch(data.x[batch], data.y[batch])[1]
        local_fitness += hist

    local_fitness /= 50
    val_ind = np.random.randint(0, len(data.x_test), 1500)

    if i % 20 == 0:
        prediction = m.predict(data.x_test[val_ind])
        hit = 0
        miss = 0
        for p, t in zip(prediction, data.y_test[val_ind]):
            if np.argmax(p) == np.argmax(t):
                hit += 1
            else:
                miss += 1
        val_acc = hit / (hit + miss)

        val_hist.append(val_acc)
    tra_hist.append(local_fitness)


    plt.scatter(i, local_fitness, color='red')
    if i % 20 == 0: plt.scatter(i, val_acc, color='blue')
    plt.legend(['training accuracy', 'validation accuracy'])
    plt.pause(0.05)

    print('Iteration', i, ': T_acc:', local_fitness, '  V_acc:', val_acc, '\t\t', i*50, 'minibatches of 16')


#history = m.fit(data.x[:N], data.y[:N], batch_size=16, epochs=1, verbose=1,
#                validation_data=(data.x_test[:val_N], data.y_test[:val_N]))