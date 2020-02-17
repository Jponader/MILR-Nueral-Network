import tensorflow as tf
from tensorflow import keras
import numpy as np

from MILR.Layers.layerNode import layerNode

class inputLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(inputLayer,self).__init__(layer, prev = prev, next = next)

