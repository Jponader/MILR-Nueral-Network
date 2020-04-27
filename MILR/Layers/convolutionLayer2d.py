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

		if not np.allclose(checkdata, self.partialData, atol=1e-08):
			checkpointed, self.biasError = biasLayer.partialCheckpoint(self)
			return self.checkpointed, True

		if self.Tlayer.use_bias:
			checkpointed, self.biasError = biasLayer.partialCheckpoint(self)
			return self.checkpointed, self.biasError 

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

		M = self.keys['M']
		Z = self.keys['Z']
		F = self.keys['F']
		FFZ = F*F*Z
		N = self.keys['N']
		Y = self.keys['Y']
		S = self.Tlayer.strides[0]

		print(outputs.shape)

		#CRC Solving
		if self.CRC:
			weightCopy = self.Tlayer.get_weights()[0]
			#weightCopy = weightCopy[:]
		#VALIDATION INSERT ERRORS
			print(weightCopy.shape)
			weightCopy[0][0][0][0] = 1
			weightCopy[2][1][0][2] = weightCopy[0][0][0][0]*-1
			weightCopy[0][2][0][1] = weightCopy[0][0][0][0]*-1
			weightCopy[2][0][0][0] = weightCopy[0][0][0][0]*-1
			weightCopy[0][1][0][0] = 2
			weightCopy[0][1][0][0] = 3


			checkCRC = []
			for f1 in weightCopy:
				for f2 in f1:
					checkCRC.append(self.CRC2D(f2))

			loc = []
			for data1, data2 in zip(checkCRC, self.store[0]):
				loc.append(self.CRC2dErrorFinder(data1,data2))

			print("ERRORS",loc)
			
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
				for j in range(varCount[i]):
					local = []
					for k in range(varCount[i]):
						print(i,j,k)
						print(ep[i][k])
						print(int(math.floor((j*S))/N) + ep[i][k][0])
						print((j*S)%N + ep[i][k][1])
						print(ep[i][k][2])

						local.append(inputs[0][int(math.floor((j*S))/N) + ep[i][k][0]]   [(j*S)%N + ep[i][k][1]]  [ep[i][k][2]])
					answers.append(local)

				print("begin",np.array(answers).shape)
				print("begin",np.array(answers))
				print(outputs[0,:,:,i].flatten()[:varCount[i]].shape)
				print(outputs[0,:,:,i].flatten()[:varCount[i]])
				
				#sol = np.linalg.solve(np.array(answers),outputs[0][:,:,i].flatten()[:varCount[i]])
				if varCount[i] == 1:
					print(outputs[0][:,:,i].flatten()[0])
					print(answers[0][0])
					sol = [outputs[0][:,:,i].flatten()[0]/answers[0][0]]
					sol2 = np.linalg.lstsq(np.array(answers),outputs[0][:,:,i].flatten()[:varCount[i]], rcond=-.1)[0]
					if sol[0] == sol2[0]:
						print("equal")
				else:
					sol = np.linalg.lstsq(np.array(answers),outputs[0][:,:,i].flatten()[:varCount[i]], rcond=-1)[0]

				#inputPrime = np.linalg.pinv(np.array(answers))
				#sol = inputPrime.dot(outputs[0][:,:,i].flatten()[:varCount[i]])

				for j in range(varCount[i]):
					print(self.Tlayer.get_weights()[0][ep[i][j][0]][ep[i][j][1]][ep[i][j][2]][i], "-----", sol[j])
					weightCopy[ep[i][j][0]][ep[i][j][1]][ep[i][j][2]][i] = sol[j]



			assert np.allclose(np.array(self.Tlayer.get_weights()[0]), weightCopy, atol=1e-2), "Kernel CRC not the same"

			return

		# Take into consideration additional data being stored, for layers that done meet the requirments
		inputs = inputs[0]
		if self.Tlayer.padding.upper() == "SAME":
			padding = (((N-1)*S)+F-N)
			left = int(math.floor(padding/2))
			right = int(math.ceil(padding/2))
			#Left Right Pad
			print(inputs.shape)
			inputs = tf.concat([tf.zeros((left,M,Z)),inputs,tf.zeros((right,M,Z))], 0)
			print(inputs.shape)
			#Top Bottom pad
			inputs = tf.concat([tf.zeros((M+padding,left,Z)),inputs,tf.zeros((M+padding,right,Z))], 1)
			print(inputs.shape)
		
		inputMatrix = []
		inputs = np.array(inputs)

		subSol = []
		k = self.Tlayer.kernel[:,:,:,0]
		I = inputs[:,:]

		for i in range(N):
			for j in range(N):
				inputMatrix.append(inputs[i*S:(i*S)+F,j*S:(j*S)+F,:].flatten())
				"""
				ConvTotal = ((I[0+(i*S)][0+(j*S)]*k[0][0])+(I[0+(i*S)][1+(j*S)]*k[0][1])+(I[0+(i*S)][2+(j*S)]*k[0][2])+
					(I[1+(i*S)][0+(j*S)]*k[1][0])+(I[1+(i*S)][1+(j*S)]*k[1][1])+(I[1+(i*S)][2+(j*S)]*k[1][2])+
					(I[2+(i*S)][0+(j*S)]*k[2][0])+(I[2+(i*S)][1+(j*S)]*k[2][1])+(I[2+(i*S)][2+(j*S)]*k[2][2]))
				subSol.append(np.sum(ConvTotal))
				"""


#theoretical solution to be tested
		inputMatrix = np.array(inputMatrix)
		outputs = testOut = np.array(outputs[0])
		outputs = np.reshape(outputs,(N*N,Y))

		print(inputMatrix.shape)
		print(outputs.shape)

		#testSol = np.reshape(scipy.linalg.solve( inputMatrix[:FFZ].astype('float64'), outputs[:FFZ,:].astype('float64')*1000000),(F,F,Z,Y))/1000000
		#testSol = scipy.linalg.lstsq( inputMatrix.astype('float64'), outputs.astype('float64'))[0]
		#testSol = np.reshape(scipy.linalg.lstsq( inputMatrix.astype('float64'), outputs.astype('float64')*1000000)[0],(F,F,Z,Y))/1000000


		inputPrime = np.linalg.pinv(inputMatrix)
		testSol = inputPrime.dot(outputs)

		testSol = np.reshape(testSol,(F,F,Z,Y))

		print("++++++")
		print(testSol)
		print("+***+")
		print(self.Tlayer.kernel[:,:,:,:])

		checkOut = self.Tlayer._convolution_op(self.rawIn, testSol)
		assert np.allclose(self.rawOut, checkOut, atol=1e-4), "Kernel Check Output not the same"

		assert np.allclose(testSol, self.Tlayer.get_weights()[0], atol=1e-4), "Kernel not the same"


	def backwardPass(self, outputs):
		#self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}
		F = self.keys['F']
		Z = self.keys['Z']
		M = self.keys['M']
		N = self.keys['N']
		Y = self.keys['Y']
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

		
		out = []
		for i in range(M):
			for j in range(M):
				print(filterMatrix[:FFZ,:FFZ].shape)
				print(outputs[0,i,j,:FFZ].shape)
				out.append(np.linalg.solve(filterMatrix[:FFZ,:FFZ],outputs[0,i,j,:FFZ]))
		
		#Possible simplification?????
		#out = (np.linalg.solve(filterMatrix[:FFZ,:FFZ],np.reshape(outputs[0],(N*N,Y))[:,:FFZ]))
		out = np.array(out)
		return [np.reshape(out,(M,M,Z))]

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
		
		#VALIDATION to be Removed
		#self.CRC = True
		#self.padded = CN.NONE

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
			self.store[0] = out

#Input Padding
		if self.padded == CN.BOTH or self.padded == CN.INPUTPAD:
			extraFilters = self.seededRandomTensor((F,F,Z,yPad))
			extraFiltered = tf.nn.conv2d(inputData, extraFilters, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)
			self.store[1]  =extraFiltered

		outputs = tf.nn.conv2d(inputData, layer.kernel, layer.strides, layer.padding.upper(), dilations=layer.dilation_rate)

# Print Summary
		print("	padded:", self.padded)
		print("	CRC:", self.CRC)
		print('	total Cost', self.cost())

		self.keys ={'F':F, 'Z':Z, 'M':M, 'N':N, 'Y':Y, 'yPad':yPad, 'mPad':mPad}
		outputs = layer._convolution_op(inputData, layer.kernel)


# Validation to be Removed
		
		#Accuracy is low, finds place it doenst work
		#rekernel = self.kernelSolver(self.rawIn, self.rawOut)


	#To Re-check/do
		"""
		if skipKernel:
			rekernel = self.backwardPass(outputs)
			assert np.allclose(rekernel, self.rawIn, atol=1e-4), "backward pass recovery"
		"""
		
		self.rawbiasIn = outputs

		if layer.use_bias:
			if layer.data_format == 'channels_first':
				outputs, status = biasLayer.layerInitilizer(self, outputs, self.Tlayer.get_weights()[1], status, data_format='NCHW')
			else:
				outputs, status = biasLayer.layerInitilizer(self, outputs, self.Tlayer.get_weights()[1], status, data_format='NHWC')
			biasLayer.kernelSolver(self, self.rawbiasIn, outputs)
			biasLayer.backwardPass(self, outputs)

		return activationLayer.staticInitilizer(outputs, layer.activation, status)


	def cost(self):

		#to add Partial Cost

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

