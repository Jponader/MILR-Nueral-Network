from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


from MILR.Layers.activationLayer import activationLayer
from MILR.Layers.biasLayer import forwardPass as biasLayer
from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

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
		#config = layer.get_config()
		#self.units = config['units']
		#Common
		#self.hasBias = config['use_bias']
		#self.activationFunc = config['activation']


	def layerInitilizer(self, inputData, status):
		return self.forwardPass(inputData, status)

	def forwardPass(self, inputs, status):
		# This function is based off of Keras.Dense.Call()
		layer = self.Tlayer
		inputs = math_ops.cast(inputs, layer._compute_dtype)
		outputs = gen_math_ops.mat_mul(inputs, layer.kernel)

		status = self.metadataHarvester(status)

		if layer.use_bias:
			outputs, status = biasLayer(outputs, layer.bias, status)

		return activationLayer.forwardPass(outputs, layer.activation, status)

	#To be updated		
	def metadataHarvester(self, status):
		if status == STAT.NO_INV:
			return STAT.REQ_INV
	
		print("			Possible Checkpoint")
		return STAT.REQ_INV
