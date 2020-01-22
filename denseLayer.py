from biasLayer import biasLayer
from activationLayer import activationLayer


import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest


class denseLayer:

	def __init__(self, weights, bias):
		print("New Dense Layer")
		self.weights = weights
		self.bias = biasLayer(bias)

	def forwardPass(self, inputMat):
		out = np.matmul(inputMat, self.weights)
		return self.bias.forwardPass(out)


	def milrIntilization(self, inputMat, state):
		#find padded output