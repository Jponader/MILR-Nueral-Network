import numpy as np


class flattenLayer:

	def __init__(self):
		print("New Flatten Layer")

	def compile(self, inputSize):
		self.inputSize = inputSize
		return(1,inputSize[0] ** len(inputSize))

	def forwardPass(self, inputMat):
		return inputMat.flatten()

	def backwardPass(self, inputMat):
		inputMat.reshape(self.inputSize)
