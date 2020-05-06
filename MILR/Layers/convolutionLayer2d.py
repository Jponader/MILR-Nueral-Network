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

class convolutionLayer2d(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(convolutionLayer2d,self).__init__(layer, prev = prev, next = next)

	def partialCheckpoint(self):
		partailInput = self.seededRandomTensor((1,*self.Tlayer.input_shape[1:]))
		checkdata  = tf.nn.conv2d(partailInput, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper())[0,0,:]
		layerError = not np.allclose(checkdata, self.partialData, atol=1e-08)

		print(self, layerError)

		self.biasError = False
		doubleError = False

		if self.Tlayer.use_bias:
			checkpointed, self.biasError = biasLayer.partialCheckpoint(self)
			print("bias-   ",self.biasError)

			if layerError == True and self.biasError ==True:
				self.biasError = False
				doubleError = True	 	

		return self.checkpointed, layerError or self.biasError, doubleError

	def forwardPass(self, inputs):
		layer = self.Tlayer
		outputs  = tf.nn.conv2d(inputData, layer.kernel, layer.strides, layer.padding.upper())

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs = biasLayer.forwardPass(outputs, layer.bias, data_format='NCHW')
			else:
				outputs = biasLayer.forwardPass(outputs, layer.bias, data_format='NHWC')

		return activationLayer.forwardPass(outputs, layer.activation)

	def kernelSolver(self, inputs, outputs):
		ogWeights = self.Tlayer.get_weights()

		if self.Tlayer.use_bias:
			if self.biasError:
				inputs = tf.nn.conv2d(inputs, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper())
				ogWeights[1] = biasLayer.kernelSolver(self, inputs, outputs)
				self.Tlayer.set_weights(ogWeights)
				return 
		
			outputs = biasLayer.backwardPass(self, outputs, data_format = self.Tlayer.data_format)

		#self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}
		M = self.keys['M']
		Z = self.keys['Z']
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

			for i in range(Y):
				if varCount[i] == 0:
					continue

				answers = []
				for j in range(min(varCount[i],N*N)):
					#print(varCount[i])
					#print(N*N)
					local = []
					for k in range(varCount[i]):
						#print(ep)
						#print(ep[i])
						#print(ep[i][k])
						#print(int(math.floor((j*S))/N) + ep[i][k][0] ,(j*S)%N + ep[i][k][1],  ep[i][k][2])
						local.append(inputs[0][int(math.floor((j*S))/N) + ep[i][k][0]]   [(j*S)%N + ep[i][k][1]]  [ep[i][k][2]])
					answers.append(local)
				
				#sol = np.linalg.solve(np.array(answers),outputs[0][:,:,i].flatten()[:varCount[i]])
			
				sol = np.linalg.lstsq(np.array(answers),outputs[0][:,:,i].flatten()[:varCount[i]], rcond=-1)[0]

				for j in range(varCount[i]):
					weightCopy[ep[i][j][0]][ep[i][j][1]][ep[i][j][2]][i] = sol[j]

			#assert np.allclose(np.array(self.Tlayer.get_weights()[0]), weightCopy, atol=1e-2), "Kernel CRC not the same"
			ogWeights[0] = weightCopy
			self.Tlayer.set_weights(ogWeights)
			return 
	#NON-CRC
# Take into consideration additional data being stored, for layers that done meet the requirments
		inputs = inputs[0]
		if self.Tlayer.padding.upper() == "SAME":
			padding = (((N-1)*S)+F-N)
			left = int(math.floor(padding/2))
			right = int(math.ceil(padding/2))
			#Left Right Pad
			inputs = tf.concat([tf.zeros((left,M,Z)),inputs,tf.zeros((right,M,Z))], 0)
			#Top Bottom pad
			inputs = tf.concat([tf.zeros((M+padding,left,Z)),inputs,tf.zeros((M+padding,right,Z))], 1)
		
		inputMatrix = []
		inputs = np.array(inputs)

		subSol = []
		k = self.Tlayer.kernel[:,:,:,0]
		I = inputs[:,:]

		for i in range(N):
			for j in range(N):
				inputMatrix.append(inputs[i*S:(i*S)+F,j*S:(j*S)+F,:].flatten())

		outputs = np.array(outputs[0])
		outputs = np.reshape(outputs,(N*N,Y))
	
		if self.padded == CN.BOTH or self.padded == CN.WEIGHTPAD:
			mPad = inputSize 
			newInput = np.array(self.seededRandomTensor((1,self.keys['mPad'],self.keys['mPad'],Z)))
			n2 = self.store[0].shape[1]

			for i in range(n2):
				for j in range(n2):
					inputMatrix.append(newInput[i*S:(i*S)+F,j*S:(j*S)+F,:].flatten())

			copyOut = np.reshape(self.store[0],(n2*n2,Y))
			outputs = np.concatenate((outputs,copyOut))

		inputMatrix = np.array(inputMatrix)
		sol = np.linalg.lstsq(inputMatrix, outputs, rcond=-1)[0]
		sol = np.reshape(sol,(F,F,Z,Y))

		#Validation Data to be Removed
		#assert np.allclose(sol, np.array(self.Tlayer.get_weights()[0]), atol=1e-4), "Kernel Solver Update"

		ogWeights[0] = sol
		self.Tlayer.set_weights(ogWeights)
		return True

#SAME PADDING
	#Insert weights sets where some are Zeros
	def backwardPass(self, outputs):
		layer = self.Tlayer

		if self.checkpointed:
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

		stride = layer.strides

		filterMatrix = []
		weights = np.array(layer.get_weights()[0])
		outputs = np.array(outputs)
		for i in range(Y):
			filterMatrix.append(weights[:,:,:,i].flatten())

		if self.padded == CN.INPUTPAD or self.padded == CN.BOTH:
			for filters in np.array(self.seededRandomTensor((F,F,Z,yPad))).T:
				filterMatrix.append(filters.flatten())

			outputs = tf.concat([outputs, self.store[1]], 3)

			
		filterMatrix = np.array(filterMatrix)
		outMatrix = np.array(outMatrix)

		out = []

		for i in range(N):
			for j in range(N):
				out.append(np.reshape(np.linalg.solve(filterMatrix,outputs[0,i,j,:]),(F,F,Z)))

		inMat = np.zeros((M,M,Z))

		for i in range(0,M-F+1,stride):
			for j in range(0,M-F+1,stride):
				inMat[i:i+F, j:j+F] = out[i*(M-F+1) + j]

		return inMat

	def layerInitilizer(self, inputData, status):
		layer = self.Tlayer

		partailInput = self.seededRandomTensor((1,*layer.input_shape[1:]))
		self.partialData = tf.nn.conv2d(partailInput, layer.kernel, layer.strides, layer.padding.upper())[0,0,:]

#Validation Data to be Removed
		#if status == STAT.NO_INV:
		#	skipKernel = False
		#else:
		#	skipKernel = True

		#self.rawIn = inputData
		#self.rawKernel = self.Tlayer.kernel
		#self.rawOut = tf.nn.conv2d(inputData, self.Tlayer.kernel, self.Tlayer.strides, self.Tlayer.padding.upper())
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

		# N needs to be set based on padding type for convolution, to be addressed
		"""
		if layer.padding.upper() == "SAME":
			N = M
		else:
			N = int(((M-F)/layer.strides[0])+1)
		"""

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
		print('	Weights: ',layer.kernel.shape)
		print("	padded:", self.padded)
		print("	CRC:", self.CRC)
		#print('	total Cost', self.cost())
#_____________

# Validation to be Removed
		
		#Accuracy is low, finds place it doenst work
		#self.biasError = False
		#rekernel = self.kernelSolver(self.rawIn, self.rawOut)
		#print("KERNEL SOLVER!!!")

		#if skipKernel:
			#rekernel = self.backwardPass(outputs)
			#assert np.allclose(rekernel, self.rawIn, atol=1e-4), "backward pass recovery"
			#print("Backward Pass Completed")
#_____________
		
		self.rawbiasIn = outputs

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs, status = biasLayer.layerInitilizer(self, outputs, self.Tlayer.get_weights()[1], status, data_format='NCHW')
			else:
				outputs, status = biasLayer.layerInitilizer(self, outputs, self.Tlayer.get_weights()[1], status, data_format='NHWC')
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

