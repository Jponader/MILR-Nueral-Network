import tensorflow as tf
from tensorflow import keras
import numpy as np

from MILR.Layers.layerNode import layerNode
import MILR.status as STAT

class zeroPaddingLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(zeroPaddingLayer,self).__init__(layer, prev = prev, next = next)
		#self.padding = layer.get_config()['padding']


	def layerInitilizer(self, inputData, status):
		out = self.Tlayer.call(inputData)
		return out, status
	