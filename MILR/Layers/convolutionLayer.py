import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest

from MILR.Layers.layerNode import layerNode



class convolutionLayer(layerNode):

	def __init__(self, filterCount, filterSize,stride = 1, activationFunc = None):
		print("New Convolution Layer")
		self.filterCount = filterCount
		self.filterSize = filterSize
		self.stride = stride

	def initalize(self, inputshape):
		#to be done calculate aspect size and generate sizes
		# return output size
		return inputshape

	def forwardPass(self, inputMat):
		out = []

		for filter in self.filters:
			out.append(Convolution(inputMat, filter, self.stride))

		out = np.array(solution)
		out =  self.bias.forwardPass(out)
		out = self.activationLayer.forwardPass(out)
		return out


	def Convolution(inputMat, filter, stride):
		inputlen = len(inputMat)
		filterLen = len(filter)
		#output = ((inputSize - filterSize + 2padding)/stride )+ 1)
		interstepSize = int(((inputlen - filterLen)/stride)+1)
		solution = np.zeros((interstepSize,interstepSize))
		sub = 0
		fil = filter.flatten()

		for x in range(0,interstepSize):
			for y in range(0,interstepSize):
				for j in range(0,filterLen):
					for i in range(0,filterLen):
						sub = (fil[(j*filterLen)+i] * inputMat[(x*stride)+j][(y*stride)+i]) + sub
				solution[x][y] = sub
				sub = 0

		return solution
		