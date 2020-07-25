import numpy as np
from tensorflow.python.ops import nn
import tensorflow as tf

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

# Open fucntions to seperate activation fucntions from other layers

def forwardPass(data, bias, data_format = None):
	return nn.bias_add(data, bias, data_format)

def partialCheckpoint(self):
	check = tf.math.reduce_sum(self.Tlayer.bias)
	return False, check !=self.biasCheck

def kernelSolver(self, inputs, outputs, data_format = None):
	length = len(inputs.shape)
	kern = outputs - inputs

	for i in range(length-1):
		kern = kern[0]

	return kern

def backwardPass(self, outputs,data_format = None):
	if data_format is not None and data_format == 'channels_first':
		inputs = outputs - tf.reshape(self.Tlayer.bias,len(self.Tlayer.bias))
	else:
		inputs = outputs - self.Tlayer.bias

	return inputs

def layerInitilizer (self, data, status, data_format=None):

	#Partial Checkpoint
	self.biasCheck = tf.math.reduce_sum(self.Tlayer.bias)
	#print("Bias Check")
	#print(self.biasCheck)
	#self.rawbias = bias

	return nn.bias_add(data, self.Tlayer.bias, data_format), STAT.REQ_INV

def cost(self):
	# Cost of partial Checkpoint
	return 0, 1, 0
