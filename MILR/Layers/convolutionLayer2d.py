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
		

	def layerInitilizer(self, inputData, status):
		self.rawIn = inputData
		self.rawKernel = self.Tlayer.kernel
		outputs = self.Tlayer._convolution_op(inputData, self.Tlayer.kernel)
		self.rawOut = outputs

		layer = self.Tlayer

		print(inputData.shape)
		print(layer.kernel.shape)
		assert len(inputData.shape) == 4, "Error: Dense Input Not 3D"
		assert len(layer.kernel.shape) == 4, "Error: Dense Kernel Not 3D"

		M = inputData.shape[1]
		Z = inputData.shape[3]
		F = layer.kernel.shape[0]
		Y = layer.kernel.shape[3]
		N = int(((M-F)/layer.strides[0])+1)
		FFZ = F*F*Z
		self.padded = CN.NONE

		if N*N < FFZ:
			nPad = math.ceil(math.sqrt(FFZ))
			self.padded = CN.WEIGHTPAD
		else:
			nPad = N
		
		inputCost = 0

		if status == STAT.NO_INV:
			status =  STAT.REQ_INV
		else:
			if Y < FFZ:
				yPad = FFZ -Y
				inputCost = yPad * N**2
				if (M*M*Z) < inputCost:
					self.checkpoint(inputData)
				else:
					if self.padded == CN.WEIGHTPAD:
						self.padded = CN.BOTH
					else:
						self.padded = CN.INPUTPAD	

		weightCost = nPad**2 - (N*N)
		print("npad", nPad)
		print("N", N)
		print("weightCost",weightCost)
		print("inputCost", inputCost)
		print(self.checkpointed)
		print(self.padded)

		if self.padded == CN.BOTH or self.padded == CN.WEIGHTPAD:
			inputSize = ((nPad -1)*layer.strides[0]) + F
			mPad = inputSize - M
			np.random.seed(self.seeder())
			pad1 =  tf.convert_to_tensor(np.random.rand(1,mPad,M,Z),  dtype= self.Tlayer.dtype)
			paddedInput = tf.concat([inputData,pad1], 1)
			pad2 =  tf.convert_to_tensor(np.random.rand(1,M+mPad,mPad,Z),  dtype= self.Tlayer.dtype)
			paddedInput = tf.concat([paddedInput,pad2], 2)
			print("weightpad",paddedInput.shape)
			paddedOut = tf.nn.conv2d(paddedInput, layer.kernel, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)
			print(paddedOut.shape)

		if self.padded == CN.BOTH or self.padded == CN.INPUTPAD:
			extraFilters = self.seededRandomTensor((F,F,Z,yPad))
			print(extraFilters.shape)
			extraFiltered = tf.nn.conv2d(inputData, extraFilters, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)
			print(extraFiltered.shape)
		
		outputs = tf.nn.conv2d(inputData, layer.kernel, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)

		assert np.allclose(self.rawOut, outputs,  atol=1e-08), "out wrong"

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

