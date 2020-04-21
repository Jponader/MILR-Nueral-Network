import math
import numpy as np
import tensorflow as tf
from enum import Enum
from random import seed
from random import randint
from random import random
from datetime import datetime


from MILR.Layers.activationLayer import activationLayer
from MILR.Layers import biasLayer
from MILR.Layers.layerNode import layerNode
from MILR.status import status as STAT

class convolutionLayer2d(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(convolutionLayer2d,self).__init__(layer, prev = prev, next = next)
		#config = layer.get_config()
		#print(config)
		#self.stride = config['strides']
		#self.filters = config['filters']
		#self.kernelSize = config['kernel_size']
		#self.padding = config['padding']
		#self.dilation = config['dilation_rate']
		#self.format = config['data_format']
		#Common
		#self.hasBias = config['use_bias']
		#self.activationFunc = config['activation']
		
#to be configured
	def partialCheckpoint(self):
		#CheckPoint, Error
		return self.checkpointed,False

	#To add partial checkpoint
	def layerInitilizer(self, inputData, status):
		layer = self.Tlayer

		print('	Weights: ',layer.kernel.shape)
		assert len(inputData.shape) == 4, "Error: Dense Input Not 3D"
		assert len(layer.kernel.shape) == 4, "Error: Dense Kernel Not 3D"

		M = inputData.shape[1]
		Z = inputData.shape[3]
		F = layer.kernel.shape[0]
		Y = layer.kernel.shape[3]

		# N needs to be set based on padding type for convolution, to be addressed
		if layer.padding.upper() == "SAME":
			N = M
		else:
			N = int(((M-F)/layer.strides[0])+1)

		FFZ = F*F*Z

		self.padded = CN.NONE
		self.CRC = False
		self.store = [None,None,None]

		nPad = N

		if N*N < FFZ:
			nPad = math.ceil(math.sqrt(FFZ))
			weightCost = (nPad**2 - (N**2))*Y

			if weightCost <= FFZ*Y/2:
				self.padded = CN.WEIGHTPAD
			else:
				self.CRC = True
		
		inputCost = 0

		if status == STAT.NO_INV:
			status =  STAT.REQ_INV
		elif F < layer.strides[0]:
			self.checkpoint(inputData)
		elif Y < FFZ:
			yPad = FFZ -Y
			inputCost = yPad * N**2
			if (M*M*Z) < inputCost:
				self.checkpoint(inputData)
			elif self.padded == CN.WEIGHTPAD:
				self.padded = CN.BOTH
			else:
				self.padded = CN.INPUTPAD	

		if self.padded == CN.BOTH or self.padded == CN.WEIGHTPAD:
			if layer.padding.upper() == "SAME":
				inputSize =  nPad
			else:
				inputSize = ((nPad -1)*layer.strides[0]) + F

			mPad = inputSize - M
			print('	MPAD: ', mPad)
			np.random.seed(self.seeder())
			pad1 =  tf.convert_to_tensor(np.random.rand(1,mPad,M,Z),  dtype= self.Tlayer.dtype)
			paddedInput = tf.concat([inputData,pad1], 1)
			pad2 =  tf.convert_to_tensor(np.random.rand(1,M+mPad,mPad,Z),  dtype= self.Tlayer.dtype)
			paddedInput = tf.concat([paddedInput,pad2], 2)
			paddedOut = tf.nn.conv2d(paddedInput, layer.kernel, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)
			print(paddedOut.shape)
			self.store[0] = (paddedOut[:,N:],paddedOut[:,:N,N:])
			print(self.store[0])


		if self.padded == CN.BOTH or self.padded == CN.INPUTPAD:
			extraFilters = self.seededRandomTensor((F,F,Z,yPad))
			extraFiltered = tf.nn.conv2d(inputData, extraFilters, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)
			self.store[1]  =extraFiltered

		if self.CRC:
			self.store[2] = None
			pass

		#Store the additional data

		outputs = tf.nn.conv2d(inputData, layer.kernel, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs, status = biasLayer.layerInitilizer(outputs, layer.bias, status, data_format='NCHW')
			else:
				outputs, status = biasLayer.layerInitilizer(outputs, layer.bias, status, data_format='NHWC')

		return activationLayer.staticInitilizer(outputs, layer.activation, status)

	def forwardPass(self, inputs):
		layer = self.Tlayer
		outputs = layer._convolution_op(inputs, layer.kernel)

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs = biasLayer.forwardPass(outputs, layer.bias, data_format='NCHW')
			else:
				outputs = biasLayer.forwardPass(outputs, layer.bias, data_format='NHWC')

		return activationLayer.forwardPass(outputs, layer.activation)


class CN(Enum):
	NONE = -1

	# yPad
	INPUTPAD = 0
		
	# nPad to derive mPad
	WEIGHTPAD = 1

	# both
	BOTH = 100

