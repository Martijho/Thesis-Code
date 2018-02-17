from dataprep import DataPrep
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Flatten
import tensorflow as tf
from keras import backend as K
from __future__ import print_function

import os
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=4))))

def task(i, x_data, y_data, verbose=True):
    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    inp = Input([28, 28, 1])
    x = Conv2D(3, (3, 3), activation='relu')(inp)
    x = Conv2D(3, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    hist = model.fit(x_data, y_data, batch_size=500, epochs=1, verbose=verbose, validation_split=0.2).history

    return hist

class myThread (threading.Thread):
   def __init__(self, threadID, x, y):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.x = x
      self.y = y
   def run(self):
      print("Starting ", self.threadID)
      task(self.threadID, self.x, self.y)
      print("Exiting ", self.threadID)



d = DataPrep()
d.mnist()



while True:
    thread1 = myThread(0, d.x, d.y)
    thread1.start()



