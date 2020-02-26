from tensorflow.python.keras import activations

import math

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

class activationLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(activationLayer,self).__init__(layer, prev = prev, next = next)
		#self.func = layer.get_config()['activation']

	def layerInitilizer(self, inputData, status):
		out = self.Tlayer.call(inputData)
		return out, status

# Open fucntions to seperate activation fucntions from other layers
	@staticmethod
	def metadataHarvester(status):
		#this needs to be updated to accoutn for all the possible activations functions and handle each type differently
		return status

	@staticmethod
	def forwardPass(data, func, status):
		return activations.get(func)(data), activationLayer.metadataHarvester(status)
