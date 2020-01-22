from modelController import modelController

from denseLayer import denseLayer
from flattenLayer import flattenLayer
from biasLayer import biasLayer
from activationLayer import activationLayer

import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest


maxMatrix = 4

seed(datetime.now())
M = randint(1,maxMatrix)
N = randint(1,maxMatrix)
P = randint(1,maxMatrix)

inputMat = np.random.rand(28,28)

weights1 = np.random.rand(784, 128) 
bias1 = np.random.rand(128)
weights2 = np.random.rand(128,10) 
bias2 = np.random.rand(10)



model = modelController()
model.add(flattenLayer())
model.add(denseLayer(weights1, bias1))
model.add(denseLayer(weights2, bias2))
out = model.forwardPass(inputMat)

print(out)
