import numpy as np
from tensorflow.python.ops import nn

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

# Open fucntions to seperate activation fucntions from other layers

def forwardPass(data, bias, data_format = None):
	return nn.bias_add(data, bias, data_format)

#to be configured
def partialCheckpoint(self):
	check = sum(bias)
	return False, check !=self.biasCheck

def kernelSolver(self, inputs, outputs):
	length = len(inputs.shape)
	kern = outputs - inputs
	print(kern.shape)
	print(kern)
	print(self.rawbias.shape)
	print(self.rawbias)
	assert np.allclose(self.rawbias, kern, atol=1e-2),"RAW BIAS"

def backwardPass(self, outputs):
	pass

def layerInitilizer (self, data, bias, status, data_format = None):
	#print(bias)
	#print(bias.shape)
	self.biasCheck = sum(bias)
	self.rawbias = bias
	print(self.biasCheck)

	#Partial Checkpoint


	return nn.bias_add(data, bias, data_format), STAT.REQ_INV

def cost(self):

		#to add Partial Cost
	return 0
