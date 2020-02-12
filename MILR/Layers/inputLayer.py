import tensorflow as tf
from tensorflow import keras
import numpy as np

from MILR.Layers.layerNode import layerNode

class inputLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(inputLayer,self).__init__(layer, prev = prev, next = next)
		self.inputShape = layer.get_config()['batch_input_shape'][1:]
		#print(self.inputShape)

	def getShape(self):
		return self.inputShape

