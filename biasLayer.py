import numpy as np

class biasLayer:

	def __init__(self, bias):
		print("New Bias Layer")
		self.bias = np.random.rand(bias)

	def forwardPass(self, inputMat):
		for i in range(0,len(self.bias)):
			inputMat[i] = inputMat[i] + self.bias[i]
		return inputMat