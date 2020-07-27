import math
import scipy
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

import time

class convolutionLayer2d(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(convolutionLayer2d,self).__init__(layer, prev = prev, next = next)

	def partialCheckpoint(self):
		partailInput = self.seededRandomTensor((1,*self.Tlayer.input_shape[1:]))
		pc = int(self.keys['N']/2)
		checkdata  = tf.nn.conv2d(partailInput, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper())[0,pc,pc,:]


		#layerError = not np.allclose(checkdata, self.partialData, atol=1e-08)
		layerError = not tf.reduce_all(tf.abs(checkdata - self.partialData) <= tf.abs(self.partialData) * 1e-5 + 1e-08)

		#print(self, layerError)

		self.biasError = False
		self.doubleError = False

		if self.Tlayer.use_bias:
			checkpointed, self.biasError = biasLayer.partialCheckpoint(self)
			#print("bias-   ",self.biasError)

			if layerError == True and self.biasError ==True:
				self.biasError = False
				self.doubleError = True	 	

		return self.checkpointed, layerError or self.biasError, self.doubleError

	def forwardPass(self, inputs):
		layer = self.Tlayer
		outputs  = tf.nn.conv2d(inputData, layer.kernel, layer.strides, layer.padding.upper())

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs = biasLayer.forwardPass(outputs, data_format='NCHW')
			else:
				outputs = biasLayer.forwardPass(outputs, data_format='NHWC')

		return activationLayer.forwardPass(outputs, layer.activation)

	def kernelSolver(self, inputs, outputs):
		ogWeights = self.Tlayer.get_weights()
		rawIn = inputs
		rawOut = outputs

		if self.Tlayer.use_bias:
			if self.biasError:
				inputs = tf.nn.conv2d(inputs, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper())
				ogWeights[1] = biasLayer.kernelSolver(self, inputs, outputs)
				self.Tlayer.set_weights(ogWeights)
				self.biasError = False
				return 
		
			outputs = biasLayer.backwardPass(self, outputs, data_format = self.Tlayer.data_format)

		#self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}
		Z = self.keys['Z']
		M = self.keys['M']
		F = self.keys['F']
		N = self.keys['N']
		Y = self.keys['Y']
		FFZ = F*F*Z
		S = self.Tlayer.strides[0]

		#CRC Solving
		if self.CRC:
			weightCopy = self.Tlayer.get_weights()[0]
	
			checkCRC = []
			for f1 in weightCopy:
				for f2 in f1:
					checkCRC.append(self.CRC2D(f2))

			loc = []
			for data1, data2 in zip(checkCRC, self.store[0]):
				loc.append(self.CRC2dErrorFinder(data1,data2))
			
			varCount = np.zeros(Y, dtype=np.int32)
			ep = []
			for i in range(Y):
				ep.append([])

			for i in range(F*F):
				for pos in loc[i]:
					varCount[pos[1]] +=1
					weightCopy[int(i/F)][i%F][pos[0]][pos[1]] = 0
					ep[pos[1]].append((int(i/F),i%F,pos[0]))

			comparison = self.Tlayer._convolution_op(inputs, weightCopy)
			outputs = np.array(outputs - comparison)

			inputs = inputs[0]
			if self.Tlayer.padding.upper() == "SAME":
				padding = (((N-1)*S)+F-N)
				left = int(math.floor(padding/2))
				right = int(math.ceil(padding/2))
				#Left Right Pad
				inputs = tf.concat([tf.zeros((left,M,Z),dtype= self.Tlayer.dtype),inputs,tf.zeros((right,M,Z),dtype= self.Tlayer.dtype)], 0)
				#Top Bottom pad
				inputs = tf.concat([tf.zeros((M+padding,left,Z),dtype= self.Tlayer.dtype),inputs,tf.zeros((M+padding,right,Z),dtype= self.Tlayer.dtype)], 1)

			for i in range(Y):
				if varCount[i] == 0:
					continue

				answers = []
				for j in range(min(varCount[i],N*N)):
					#print(varCount[i])
					#print(N*N)
					#local = []
					for k in range(varCount[i]):
						#print(ep)
						#print(ep[i])
						#print(ep[i][k])
						#print(int(math.floor((j*S))/N) + ep[i][k][0] ,(j*S)%N + ep[i][k][1],  ep[i][k][2])
						answers = tf.concat([answers,[inputs[int(math.floor((j*S))/N) + ep[i][k][0]][(j*S)%N + ep[i][k][1]][ep[i][k][2]]]],0)
					#answers = tf.concat([answers,[local]], -1)
				answers = tf.reshape(answers,(min(varCount[i],N*N),varCount[i]))
				
				#sol = np.linalg.solve(np.array(answers),outputs[0][:,:,i].flatten()[:varCount[i]])
				#sol = np.linalg.lstsq(np.array(answers),outputs[0][:,:,i].flatten()[:varCount[i]], rcond=-1)[0]

				sol = tf.linalg.lstsq(answers, tf.reshape(outputs[0][:,:,i],(-1,1))[:varCount[i]], l2_regularizer=0.0, fast=False)

				for j in range(varCount[i]):
					weightCopy[ep[i][j][0]][ep[i][j][1]][ep[i][j][2]][i] = sol[j]

			# assert np.allclose(np.array(self.Tlayer.get_weights()[0]), weightCopy, atol=1e-4), "Kernel CRC not the same"
			ogWeights[0] = weightCopy
			self.Tlayer.set_weights(ogWeights)
		else:
	#NON-CRC
			inputs = inputs[0]
			if self.Tlayer.padding.upper() == "SAME":
				padding = (((N-1)*S)+F-N)
				left = int(math.floor(padding/2))
				right = int(math.ceil(padding/2))
				#Left Right Pad
				inputs = tf.concat([tf.zeros((left,M,Z),dtype= self.Tlayer.dtype),inputs,tf.zeros((right,M,Z),dtype= self.Tlayer.dtype)], 0)
				#Top Bottom pad
				inputs = tf.concat([tf.zeros((M+padding,left,Z),dtype= self.Tlayer.dtype),inputs,tf.zeros((M+padding,right,Z),dtype= self.Tlayer.dtype)], 1)
			
			
			inputs = np.array(inputs)

			subSol = []

			inputMatrix = []
			for i in range(N):
				for j in range(N):
					#inputMatrix = tf.concat([inputMatrix,inputs[i*S:(i*S)+F,j*S:(j*S)+F,:].flatten()],0)
					#inputMatrix = tf.stack([inputMatrix,[inputs[i*S:(i*S)+F,j*S:(j*S)+F,:].flatten()]])
					inputMatrix.append(inputs[i*S:(i*S)+F,j*S:(j*S)+F,:].flatten())
			
			#outputs = np.array(outputs[0])
			outputs = tf.reshape(outputs,(N*N,Y))
		
			if self.padded == CN.BOTH or self.padded == CN.WEIGHTPAD:
				mPad = inputSize 
				newInput = np.array(self.seededRandomTensor((1,self.keys['mPad'],self.keys['mPad'],Z)))
				n2 = self.store[0].shape[1]

				for i in range(n2):
					for j in range(n2):
						inputMatrix.append(newInput[i*S:(i*S)+F,j*S:(j*S)+F,:].flatten())

				copyOut = tf.reshape(self.store[0],(n2*n2,Y))
				outputs = tf.concat([outputs,copyOut],0)

			#inputMatrix = np.array(inputMatrix)
			#sol = np.linalg.lstsq(inputMatrix, outputs, rcond=-1)[0]
			sol = tf.linalg.lstsq(inputMatrix, outputs, l2_regularizer=0.0, fast=False)
			sol = tf.reshape(sol,(F,F,Z,Y))

			#Validation Data to be Removed
			#assert np.allclose(sol, np.array(self.Tlayer.get_weights()[0]), atol=1e-2), (sol,np.array(self.Tlayer.get_weights()[0]) )

			ogWeights[0] = sol
			self.Tlayer.set_weights(ogWeights)

		if self.doubleError:
			if self.Tlayer.use_bias:
				inputs = tf.nn.conv2d(rawIn, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper())
				ogWeights[1] = biasLayer.kernelSolver(self, inputs, rawOut)
				self.Tlayer.set_weights(ogWeights)
				self.biasError = False
				self.doubleError = False
			

	def backwardPass(self, outputs):
		layer = self.Tlayer

		if self.checkpointed:
			print("Checkpoint Used")
			return self.checkpointData

		if layer.use_bias:
			outputs = biasLayer.backwardPass(self, outputs, data_format = layer.data_format)

		#self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}
		F = self.keys['F']
		Z = self.keys['Z']
		M = self.keys['M']
		N = self.keys['N']
		Y = self.keys['Y']
		yPad = self.keys['yPad']
		FFZ = F*F*Z

		stride = layer.strides[0]

		filterMatrix = []
		weights = np.array(layer.get_weights()[0])
		outputs = np.array(outputs)
		for i in range(Y):
			filterMatrix.append(weights[:,:,:,i].flatten())

		#checkdata  = tf.nn.conv2d(self.rawIn, np.reshape(filterMatrix[0], (F,F,Z,1)), self.Tlayer.strides, self.Tlayer.padding.upper())

		if self.padded == CN.INPUTPAD or self.padded == CN.BOTH:
			pad =  np.array(self.seededRandomTensor((F,F,Z,yPad)))
			for i in range(yPad):
				filterMatrix.append(pad[:,:,:,i].flatten())
			outputs = tf.concat([outputs, self.store[1]], 3)

		filterMatrix = np.array(filterMatrix)

		out = []

		for i in range(N):
			for j in range(N):
				out.append(tf.reshape(tf.linalg.solve(filterMatrix,outputs[0,i,j,:]),(F,F,Z)))

		if self.Tlayer.padding.upper() == "SAME":
			padding = (((N-1)*stride)+F-M)
			left = int(math.floor(padding/2))
			right = int(math.ceil(padding/2))
			M = M + padding

			inMat = tf.zeros((M,M,Z),dtype= self.Tlayer.dtype)

			count = 0

			for i in range(0,(M-F+1), stride):
					for j in range(0,(M-F+1),stride):
						inMat[i:i + F, j:j + F] = out[count]
						count += 1

			return tf.reshape(inMat[left:-right,left:-right], (1,self.keys['M'],self.keys['M'],Z))

		inMat = tf.zeros((M,M,Z),dtype= self.Tlayer.dtype)

		count = 0

		for i in range(0,(M-F+1), stride):
				for j in range(0,(M-F+1),stride):
					inMat[i:i + F, j:j + F] = out[count]
					count += 1

		return tf.reshape(inMat, (1,M,M,Z))

	def layerInitilizer(self, inputData, status):
		layer = self.Tlayer
		pc = int(layer.output_shape[1]/2)
		partailInput = self.seededRandomTensor((1,*layer.input_shape[1:]))
		self.partialData = tf.nn.conv2d(partailInput, layer.kernel, layer.strides, layer.padding.upper())[0,pc,pc,:]

#Validation Data to be Removed
		# print(self.partialData.shape)
		if status == STAT.NO_INV:
			skipKernel = False
		else:
			skipKernel = True

		self.rawIn = inputData
		self.rawKernel = self.Tlayer.kernel
		self.rawOut = tf.nn.conv2d(inputData, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper())
#_____________

		layer = self.Tlayer

		assert len(inputData.shape) == 4, "Error: Dense Input Not 3D"
		assert len(layer.kernel.shape) == 4, "Error: Dense Kernel Not 3D"

		M = layer.input_shape[1]
		Z = layer.input_shape[3]
		F = layer.kernel.shape[0]
		Y = layer.kernel.shape[3]
		N = layer.output_shape[1]
		NN = N*N
		FFZ = F*F*Z

		self.padded = CN.NONE
		self.CRC = False
		self.store = [None,None]

		mPad = N
		yPad = FFZ -Y
		nPad = N

#Determine Padding Type and Requirments
		
		if FFZ > 50:
			self.CRC = True
		elif N*N < FFZ:
			nPad = math.ceil(math.sqrt(FFZ-NN))
			weightCost = (nPad**2)*Y

			if weightCost <= FFZ*Y/2:
				self.padded = CN.WEIGHTPAD
			else:
				self.CRC = True
		
#VALIDATION to be Removed
		#self.CRC = True
		#self.padded = CN.NONE
#_____________

		#Invertiability Requirments
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
				mPad =  nPad
			else:
				mPad = ((nPad -1)*layer.strides[0]) + F
 
			newInput = self.seededRandomTensor((1,mPad,mPad,Z))
			self.store[0] = tf.nn.conv2d(newInput, layer.kernel, layer.strides, layer.padding.upper())
		elif self.CRC:
			out = []
			for f1 in layer.get_weights()[0]:
				for f2 in f1:
					out.append(self.CRC2D(f2))
			self.store[0] = out

		# Input Padding
		if self.padded == CN.BOTH or self.padded == CN.INPUTPAD:
			extraFilters = self.seededRandomTensor((F,F,Z,yPad))
			self.store[1] = tf.nn.conv2d(inputData, extraFilters, layer.strides, layer.padding.upper())

		outputs = tf.nn.conv2d(inputData, layer.kernel, layer.strides, layer.padding.upper())
		self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}

# Print Summary
		# print('	Weights: ',layer.kernel.shape)
		# print("	padded:", self.padded)
		# print("	CRC:", self.CRC)
		#print('	total Cost', self.cost())
#_____________

# Validation to be Removed
		
		#Accuracy is low, finds place it doenst work
		# self.biasError = False
		# rekernel = self.kernelSolver(self.rawIn, self.rawOut)
		# print("KERNEL SOLVER!!!")

		# if skipKernel:
		# 	rekernel = self.backwardPass(outputs)
		# 	# print(rekernel.shape)
		# 	# print(rekernel)
		# 	# print(self.rawIn.shape)
		# 	# print(self.rawIn)
		# 	assert np.allclose(rekernel, self.rawIn, atol=1e-2), "backward pass recovery"
		# 	print("Backward Pass Completed")

		# self.rawbiasIn = outputs
#_____________
		
		if self.CRC:
			print(self.name +"CRC")

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs, status = biasLayer.layerInitilizer(self, outputs, status, data_format='NCHW')
			else:
				outputs, status = biasLayer.layerInitilizer(self, outputs, status, data_format='NHWC')

# Validation to be Removed
			#biasLayer.kernelSolver(self, self.rawbiasIn, outputs)
			#biasLayer.backwardPass(self, outputs, data_format = layer.data_format)
#_____________

		return activationLayer.staticInitilizer(outputs, layer.activation, status)


	def cost(self):
		check, part, stored = super(convolutionLayer2d,self).cost()

		for i in self.store:
			if i == None:
				continue
			if self.CRC == True and stored == 0:
				for j in i:
					for n in j:
						hold = 1
						for n in n.shape:
							hold = hold * n
						stored += hold
			else:
				hold = 1
				for j in i.shape:
					hold = hold * j
				stored += hold

		if self.partialData is not None:
			hold  = 1
			for i in self.partialData.shape:
				hold = i * hold
			part = hold

		if self.Tlayer.use_bias:
			c, p, s = biasLayer.cost(self)
			check += c
			part += p
			stored += s

		return check, part, stored

class CN(Enum):
	NONE = -1

	# yPad
	INPUTPAD = 0
		
	# nPad to derive mPad
	WEIGHTPAD = 1

	# both
	BOTH = 100

