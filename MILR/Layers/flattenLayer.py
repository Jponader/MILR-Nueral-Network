import numpy as np

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

from tensorflow.keras import layers as L
import tensorflow as tf

class flattenLayer(layerNode):

	def __init__(self, layer, prev = None, next = None):
		super(flattenLayer,self).__init__(layer, prev = prev, next = next)

	def backwardPass(self, outputs):
		return tf.reshape(outputs, (1,*self.Tlayer.input_shape[1:]))
