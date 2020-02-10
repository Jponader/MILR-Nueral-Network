import tensorflow as tf
from tensorflow import keras
import numpy as np

from MILR.Layers.layerNode import layerNode

class inputLayer(layerNode):

	def __init__ (self,layer, prev = None):
		super(inputLayer,self).__init__(prev = prev)
		self.Tlayer = layer
		print(layer.name)
		self.name = layer.name
		#print(layer.get_config())
		self.inputShape = layer.get_config()['batch_input_shape'][1:]
		#print(self.inputShape)

	def getShape(self):
		return self.inputShape

