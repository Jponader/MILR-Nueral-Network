import numpy as np
from tensorflow.python.ops import nn


from MILR.Layers.layerNode import layerNode


class biasLayer():

	def __init__(self):
		pass



# Open fucntions to seperate activation fucntions from other layers

	def forwardPass(data, bias, data_format = None):
		return nn.bias_add(data, bias, data_format)
