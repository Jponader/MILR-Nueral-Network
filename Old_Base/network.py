from modelController import modelController

from denseLayer import denseLayer
from flattenLayer import flattenLayer
from biasLayer import biasLayer
from activationLayer import activationLayer as act

from tensorflowLoader import tensorflowLoader

from loss import loss
#from optomizer import optomizer

import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest

#---------------------------------------------------
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Input shape 1000,28,28
#Output shape 1000, just the correct index

#-------------------------------------------------------

print(x_test)
print(x_test.shape)
print(y_test)
print(y_test.shape)

maxMatrix = 4

seed(datetime.now())
M = randint(1,maxMatrix)
N = randint(1,maxMatrix)
P = randint(1,maxMatrix)

inputMat = np.random.rand(28,28)


model = modelController()
model.add(flattenLayer())
model.add(denseLayer(128, activationFunc = act.relu))
model.add(denseLayer(10, activationFunc = act.softmax))

model.compile((28,28), loss = loss.sparse_categorical_crossentropy)

model.train(x_test, y_test)

out = model.forwardPass(inputMat)

print(out)



loader = tensorflowLoader('model.h5')

