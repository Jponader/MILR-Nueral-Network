class biasLayer:

	def __init__(self, bias = []):
		print("New Bias Layer")
		self.bias = bias

	def forwardPass(self, inputMat):
		for i in range(0,self.bias.shape[0]):
			inputMat[i] = inputMat[i] + self.bias[i]
		return inputMat