from MILR import Layers as L

from MILR.Layers.activationLayer import activationLayer
from MILR.Layers.biasLayer import biasLayer
from MILR.Layers.layerNode import layerNode
import MILR.status as STAT


import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest


class denseLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(denseLayer,self).__init__(layer, prev = prev, next = next)
		config = layer.get_config()
		self.units = config['units']
		
		#Common
		self.hasBias = config['use_bias']
		self.activationFunc = config['activation']


	#Need to seperate out Bias
	def layerInitilizer(self, inputData, status):
		out = self.Tlayer.call(inputData)
		#print(out.shape)
		return out, status



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