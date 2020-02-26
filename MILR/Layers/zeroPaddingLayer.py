import tensorflow as tf
from tensorflow import keras
import numpy as np

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

class zeroPaddingLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(zeroPaddingLayer,self).__init__(layer, prev = prev, next = next)

