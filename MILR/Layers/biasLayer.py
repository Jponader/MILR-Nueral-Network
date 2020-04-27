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

def kernelSolver(self, inputs, outputs):
	length = len(inputs.shape)
	kern = outputs - inputs

	for i in range(length-1):
		kern = kern[0]

	#print(kern.shape)
	#print(kern)
	#print(self.rawbias.shape)
	#print(self.rawbias)
	assert np.allclose(self.rawbias, kern, atol=1e-2),"RAW BIAS"

def backwardPass(self, outputs):
	inputs = outputs - self.Tlayer.get_weights()[1]
	assert np.allclose(inputs, self.rawbiasIn, atol=1e-2), "backwardPass"

def layerInitilizer (self, data, bias, status, data_format = None):

	#Partial Checkpoint
	self.biasCheck = sum(bias)
	self.rawbias = bias

	return nn.bias_add(data, bias, data_format), STAT.REQ_INV

def cost(self):

	return 1
