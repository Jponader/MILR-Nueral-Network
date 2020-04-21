import numpy as np
from tensorflow.python.ops import nn

from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

# Open fucntions to seperate activation fucntions from other layers

def forwardPass(data, bias, data_format = None):
	return nn.bias_add(data, bias, data_format)

#to be configured
	def partialCheckpoint(self):
		#CheckPoint, Error
		return self.checkpointed,False

def layerInitilizer (data, bias, status, data_format = None):
	#To add partial Checkpoint
	return nn.bias_add(data, bias, data_format), STAT.REQ_INV
	
def kernelSolver(inputs, outputs):
	pass

def backwardPass(outputs):
	pass