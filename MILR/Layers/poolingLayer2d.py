import tensorflow as tf
from tensorflow import keras
import numpy as np

from MILR.Layers.layerNode import layerNode

class poolingLayer2d(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(poolingLayer2d,self).__init__(layer, prev = prev, next = next)
		config = layer.get_config()
		self.poolSize = config['pool_size']
		self.stride = config['strides']
