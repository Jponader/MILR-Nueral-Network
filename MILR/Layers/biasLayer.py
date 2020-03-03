import numpy as np
from tensorflow.python.ops import nn

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

# Open fucntions to seperate activation fucntions from other layers

def forwardPass(data, bias, data_format = None):
	return nn.bias_add(data, bias, data_format)


def staticInitilizer (data, bias, status, data_format = None):
	return nn.bias_add(data, bias, data_format), STAT.REQ_INV
	