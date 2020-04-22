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
		

	def partialCheckpoint(self):
		partailInput = self.seededRandomTensor((1,*self.Tlayer.input_shape[1:]))
		checkdata  = tf.nn.conv2d(partailInput, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper(), dilations=self.Tlayer.dilation_rate)[0,0,:]
		#CheckPoint, Error
		return self.checkpointed, not np.allclose(checkdata, self.partialData, atol=1e-08)

	def forwardPass(self, inputs):
		layer = self.Tlayer
		outputs = layer._convolution_op(inputs, layer.kernel)

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs = biasLayer.forwardPass(outputs, layer.bias, data_format='NCHW')
			else:
				outputs = biasLayer.forwardPass(outputs, layer.bias, data_format='NHWC')

		return activationLayer.forwardPass(outputs, layer.activation)

	def kernelSolver(self, inputs, outputs):
		#self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}
		# consider padding Same and Valid
		# Additional input data

		pass

	def backwardPass(self, outputs):
		#self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}
		F = self.keys['F']
		Z = self.keys['Z']
		M = self.keys['M']
		yPad = self.keys['yPad']
		FFZ = F*F*Z

		layer = self.Tlayer

		if self.checkpointed:
			return self.checkpointData

		filterMatrix = []
		for filters in np.array(layer.get_weights()[0]).T:
			filterMatrix.append(filters.flatten())

		if self.padded == CN.INPUTPAD or self.padded == CN.BOTH:
			for filters in np.array(self.seededRandomTensor((F,F,Z,yPad))).T:
				filterMatrix.append(filters.flatten())

			outputs = tf.concat([outputs, self.store[1]], 3)
			
		filterMatrix = np.array(filterMatrix)
		filterShape = filterMatrix.shape
		print(filterShape)
		print(outputs.shape)

		print(filterMatrix[:FFZ,:FFZ].shape)
		print(outputs[0,0,0,:FFZ].shape)

		out = []
		for i in range(M):
			for j in range(M):
				out.append(np.linalg.solve(filterMatrix[:FFZ,:FFZ],outputs[0,i,j,:FFZ]))
		
		out = np.array(out)
		return [np.reshape(out,(M,M,Z))]


	#To add partial checkpoint
	def layerInitilizer(self, inputData, status):
		# partial checkpoint
		partailInput = self.seededRandomTensor((1,*self.Tlayer.input_shape[1:]))
		self.partialData = tf.nn.conv2d(partailInput, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper(), dilations=self.Tlayer.dilation_rate)[0,0,:]

#Validation Data to be Removed
		if status == STAT.NO_INV:
			skipKernel = False
		else:
			skipKernel = True

		self.rawIn = inputData
		self.rawKernel = self.Tlayer.kernel
		self.rawOut = tf.nn.conv2d(inputData, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper(), dilations=self.Tlayer.dilation_rate)

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
		NN = N*N

		self.padded = CN.NONE
		self.CRC = False
		self.store = [None,None]

		mPad = N
		yPad = FFZ -Y
		nPad = N

#Determine Padding Type and Requirments
		if N*N < FFZ:
			nPad = math.ceil(math.sqrt(FFZ-NN))
			weightCost = (nPad**2)*Y

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
		


# Weight Padding and CRC
		if self.padded == CN.BOTH or self.padded == CN.WEIGHTPAD:
			if layer.padding.upper() == "SAME":
				inputSize =  nPad
			else:
				inputSize = ((nPad -1)*layer.strides[0]) + F

			mPad = inputSize 
			newInput = self.seededRandomTensor((1,mPad,mPad,Z))
			newOut = tf.nn.conv2d(newInput, layer.kernel, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)
			self.store[0] = (newOut)
		elif self.CRC:
			out = []
			for f1 in layer.get_weights()[0]:
				for f2 in f1:
					out.append(self.CRC2D(f2))
			self.store[0] = np.array(out)

			print(type(layer.get_weights()[0]))
			print(layer.get_weights()[0].shape)
			for f1 in layer.get_weights()[0]:
				for f2 in f1:
					out.append(self.CRC2D(f2))
			self.store[0] = np.array(out)
			print(self.store[0])

#Input Padding DONE!!!
		if self.padded == CN.BOTH or self.padded == CN.INPUTPAD:
			extraFilters = self.seededRandomTensor((F,F,Z,yPad))
			extraFiltered = tf.nn.conv2d(inputData, extraFilters, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)
			self.store[1]  =extraFiltered

		outputs = tf.nn.conv2d(inputData, layer.kernel, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)

# Print Summary
		print("	padded:", self.padded)
		print("	CRC:", self.CRC)
		print('	total Cost', self.cost())

# Print Summary
		print("	padded:", self.padded)
		print("	CRC:", self.CRC)
		print('	total Cost', self.cost())

		self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}
		outputs = layer._convolution_op(inputData, layer.kernel)


# Validation to be Removed
		if skipKernel:
			rekernel = self.backwardPass(outputs)
			print(rekernel)
			print(self.rawIn)
			assert np.allclose(rekernel, self.rawIn, atol=1e-0), "backward pass recovery"

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs, status = biasLayer.layerInitilizer(outputs, layer.bias, status, data_format='NCHW')
			else:
				outputs, status = biasLayer.layerInitilizer(outputs, layer.bias, status, data_format='NHWC')

		return activationLayer.staticInitilizer(outputs, layer.activation, status)

	def cost(self):
		total = 0
		if self.checkpointed:
			cost = 1
			for i in self.checkpointData.shape:
				cost = cost*i
			total = total + cost

		cost = 0
		for i in self.store:
			if i == None:
				continue

			if self.CRC == True and cost == 0:
				for j in i:
					for n in j:
						hold = 1
						for n in n.shape:
							hold = hold * n
						cost += hold
			else:
				hold = 1
				for j in i.shape:
					hold = hold * j
				cost += hold

		total = total + cost

		return total



	def cost(self):
		total = 0
		if self.checkpointed:
			cost = 1
			for i in self.checkpointData.shape:
				cost = cost*i
			total = total + cost

		cost = 0
		for i in self.store:
			if i == None:
				continue

			if self.CRC == True and cost == 0:
				for j in i:
					for n in j:
						hold = 1
						for n in n.shape:
							hold = hold * n
						cost += hold
			else:
				hold = 1
				for j in i.shape:
					hold = hold * j
				cost += hold

		total = total + cost

		return total


	def cost(self):
		total = 0
		if self.checkpointed:
			cost = 1
			for i in self.checkpointData.shape:
				cost = cost*i
			total = total + cost

		cost = 0
		for i in self.store:
			if i == None:
				continue

			if self.CRC == True and cost == 0:
				for j in i:
					for n in j:
						hold = 1
						for n in n.shape:
							hold = hold * n
						cost += hold
			else:
				hold = 1
				for j in i.shape:
					hold = hold * j
				cost += hold

		total = total + cost

		return total




class CN(Enum):
	NONE = -1

	# yPad
	INPUTPAD = 0
		
	# nPad to derive mPad
	WEIGHTPAD = 1

	# both
	BOTH = 100

