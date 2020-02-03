import numpy as np


class flattenLayer:

	def __init__(self):
		print("New Flatten Layer")

	def compile(self, inputSize):
		self.inputSize = inputSize
		return(inputSize[0] ** len(inputSize),1)

	def forwardPass(self, inputMat):
		return inputMat.flatten().T

	def backwardPass(self, inputMat):
		inputMat.T.reshape(self.inputSize)
