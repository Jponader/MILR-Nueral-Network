from MILR.Layers.layerNode import layerNode
import MILR.status as STAT

class addLayer(layerNode):

	def __init__ (self,layer, prev = None, next = None):
		super(addLayer,self).__init__(layer, prev = prev, next = next)
		self.forwardInputs = []
		self.inputData = []


	def initilize(self, inputSize, status = STAT.START, inputData = None):
		self.inputSize.append(inputSize)
		self.inputData.append(inputData)
		if self.canStartInilize():
			self.outputSize = self.Tlayer.compute_output_shape(self.inputSize)
			print(self, self.outputSize)

			if status == STAT.START:
				if self.inputLayer:
					inputData = self.startMetadata()
					status = STAT.NO_INV
				else:
					print("ERROR :Not Start Layer")
					sys.exit()

			if inputData is None:
				print("ERROR : No input data for next round")
				sys.exit()

			outputData, status = self.layerInitilizer(self.inputData, status)

			if not self.end:
				for n in self.next:
					n.initilize(self.outputSize, status, inputData = outputData)


	def canStartInilize(self):
		if len(self.inputSize) == len(self.prev) and len(self.inputData) == len(self.prev):
			return True
		else:
			return False


	def layerInitilizer(self, inputData, status):
		self.forwardInputs.append(inputData)
		out = self.Tlayer.call(inputData)
		print(out.shape)
		return out, status


