import numpy as np
from tensorflow.python.ops import nn

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

# Open fucntions to seperate activation fucntions from other layers

def forwardPass(data, bias, data_format = None):
	return nn.bias_add(data, bias, data_format)

def partialCheckpoint(self):
	check = sum(self.Tlayer.get_weights()[1])
	return False, check !=self.biasCheck

def kernelSolver(self, inputs, outputs, data_format = None):
	length = len(inputs.shape)
	kern = outputs - inputs

	for i in range(length-1):
		kern = kern[0]

	return kern

def backwardPass(self, outputs,data_format = None):
	if data_format is not None and data_format == 'channels_first':
		inputs = outputs - np.reshape(self.Tlayer.bias,len(self.Tlayer.bias))
	else:
		inputs = outputs - self.Tlayer.bias

	return inputs

def layerInitilizer (self, data, bias, status, data_format=None):

	#Partial Checkpoint
	self.biasCheck = sum(bias)
	#self.rawbias = bias

	return nn.bias_add(data, bias, data_format), STAT.REQ_INV

def cost(self):
	# Cost of partial Checkpoint
	return 1
