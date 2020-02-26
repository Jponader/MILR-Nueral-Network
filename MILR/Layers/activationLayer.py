from tensorflow.python.keras import activations

import math

from MILR.Layers.layerNode import layerNode
import MILR.status as STAT



class activationLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(activationLayer,self).__init__(layer, prev = prev, next = next)
		self.func = layer.get_config()['activation']

	def layerInitilizer(self, inputData, status):
		out = self.Tlayer.call(inputData)
		return out, status








# Open fucntions to seperate activation fucntions from other layers


	def forwardPass(func, data):
		results = activations.get(getattr(activations,func))(data)
		return results



