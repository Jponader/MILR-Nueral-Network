import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime

from MILR.Layers.activationLayer import activationLayer
from MILR.Layers.biasLayer import forwardPass as biasLayer
from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

class convolutionLayer2d(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(convolutionLayer2d,self).__init__(layer, prev = prev, next = next)
		#config = layer.get_config()
		#print(config)
		#self.stride = config['strides']
		#self.filters = config['filters']
		#self.kernelSize = config['kernel_size']
		#self.padding = config['padding']
		#self.dilation = config['dilation_rate']
		#self.format = config['data_format']
		#Common
		#self.hasBias = config['use_bias']
		#self.activationFunc = config['activation']
		

	def layerInitilizer(self, inputData, status):
		return self.forwardPass(inputData, status)

	def forwardPass(self, inputs, status):
		layer = self.Tlayer
		outputs = layer._convolution_op(inputs, layer.kernel)

		status = self.metadataHarvester(status)

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs, status = biasLayer(outputs, layer.bias, status, data_format='NCHW')
			else:
				outputs, status = biasLayer(outputs, layer.bias, status, data_format='NHWC')

		return activationLayer.forwardPass(outputs, layer.activation, status)

	#To be updated		
	def metadataHarvester(self, status):
		if status == STAT.NO_INV:
			return STAT.REQ_INV
	
		print("			Possible Checkpoint")
		return STAT.REQ_INV
