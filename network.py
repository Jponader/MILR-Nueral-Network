from modelController import modelController

from denseLayer import denseLayer
from flattenLayer import flattenLayer
from biasLayer import biasLayer
from activationLayer import activationLayer as act

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


model = modelController()
model.add(flattenLayer())
model.add(denseLayer(128, activationFunc = act.relu))
model.add(denseLayer(10, activationFunc = act.softmax))

model.compile((28,28))
out = model.forwardPass(inputMat)

print(out)
