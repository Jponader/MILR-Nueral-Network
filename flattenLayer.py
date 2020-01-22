


class flattenLayer:

	def __init__(self):
		print("New Flatten Layer")

	def forwardPass(self, inputMat):
		return inputMat.flatten()
