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

	def __init__(self,depth, activationFunc = None):
		print("New Dense Layer")
		self.depth = depth
		self.bias = biasLayer(depth)
		self.activationLayer = activationLayer(activationFunc)

	def compile(self, inputshape):
		if len(inputshape) > 2:
				print("Error: Input Mat not 2d")

		outputShape = (inputshape[0],self.depth )
		self.weightShape = (inputshape[1], self.depth)
		self.weights = np.random.rand(inputshape[1], self.depth)
		return outputShape


	def forwardPass(self, inputMat):
		out = np.matmul(inputMat, self.weights)
		out =  self.bias.forwardPass(out)
		out = self.activationLayer.forwardPass(out)
		return out


	#def milrIntilization(self, inputMat, state):
		#find padded output