from tensorflow.python.keras import activations

import math

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

class activationLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(activationLayer,self).__init__(layer, prev = prev, next = next)

	def layerInitilizer(self, inputData, status):
		#out = self.Tlayer.call(inputData)
		return inputData, status

# Open fucntions to seperate activation fucntions from other layers
	@staticmethod
	def metadataHarvester(status):
		#this needs to be updated to accoutn for all the possible activations functions and handle each type differently
		return status

	@staticmethod
	def forwardPass(data, func, status):
		#return activations.get(func)(data), activationLayer.metadataHarvester(status)
		return data, status

# If one treats all of the activation layers on passthrough functions, 
# they become invertible and the realtionsip between input and output for others layer is maintained
# as there are no weight and the shap does not change we can theroretically ignore this layer entirely