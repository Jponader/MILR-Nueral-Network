import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime

from MILR.Layers.activationLayer import activationLayer
from MILR.Layers.biasLayer import biasLayer
from MILR.Layers.layerNode import layerNode
import MILR.status as STAT



class convolutionLayer2d(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(convolutionLayer2d,self).__init__(layer, prev = prev, next = next)
		config = layer.get_config()
		self.stride = config['strides']
		self.filters = config['filters']
		self.kernelSize = config['kernel_size']
		#Common
		self.hasBias = config['use_bias']
		self.activationFunc = config['activation']



	#Need to seperate out Bias
	def layerInitilizer(self, inputData, status):
		out = self.Tlayer.call(inputData)
		print(out.shape)
		return out, status
	
		