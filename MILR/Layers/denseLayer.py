from MILR import Layers as L

from MILR.Layers.biasLayer import biasLayer
from MILR.Layers.layerNode import layerNode


import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest


class denseLayer(layerNode):

	def __init__(self, layer, prev = None):
		super(denseLayer,self).__init__(prev = prev)
		self.Tlayer = layer
		print(layer.name)
		self.name = layer.name
		#print(layer.get_config())
		self.units = layer.get_config()['units']
		self.use_bias = layer.get_config()['use_bias']
		self.activationFunc = layer.get_config()['activation']

	def initalize(self, inputshape):
		if len(inputshape) > 2:
				print("Error: Input Mat not 2d")

		outputShape = (self.depth, inputshape[1])
		self.weightShape = (self.depth, inputshape[0])
		self.weights = np.random.rand(self.depth, inputshape[0])
		return outputShape


	def forwardPass(self, inputMat):
		out = np.matmul(self.weights, inputMat)
		out =  self.bias.forwardPass(out)
		out = self.activationLayer.forwardPass(out)
		return out

	def backPropogation(self, gradient):
		return 0


	#def milrIntilization(self, inputMat, state):
		#find padded output