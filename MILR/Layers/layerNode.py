class layerNode:

	def __init__(self, layer, prev = None, next = None, end = False):
		self.prev = [prev]
		self.next = [next]
		self.end = end
		self.Tlayer = layer
		#print(layer.name)
		self.name = layer.name
		self.inputLayer = False
		self.inputSize = []
		self.outputSize = None

	def __str__(self):
		return self.name 
		#+ " Next " + str(len(self.next)) + " Prev " + str(len(self.prev))

	def initilize(self, inputSize):
		self.inputSize.append(inputSize)
		if self.canStartInilize():
			#self.outputSize = self.layerInitilizer(inputSize)
			if len(self.inputSize) > 1:
				self.outputSize = self.Tlayer.compute_output_shape(self.inputSize)
			else:
				self.outputSize = self.Tlayer.compute_output_shape(self.inputSize[0])
			print(self, self.outputSize)
			if not self.end:
				for n in self.next:
					n.initilize(self.outputSize)

	# Assumption of Passthrough Layers will have same input as oputput
	def layerInitilizer(self,inputSize):
		return inputSize

	def canStartInilize(self):
		if len(self.inputSize) == len(self.prev):
			return True
		else:
			return False

	def setAsInputLayer(self):
		self.inputLayer = True
		return self.Tlayer.get_config()['batch_input_shape'][1:]

	def setNext(self, next):
		if self.next[0] == None:
			self.next[0] = next
		else:
			self.next.append(next)

	def setPrev(self, prev):
		if self.prev[0] == None:
			self.prev[0] = prev
		else:
			self.prev.append(prev)

	def hasNext(self):
		return not (self.end) and self.next[0] != None

	def isEnd(self):
		self.end = True

	def getNext(self):
		return self.next

	def getPrev(self):
		return self.prev

	def isSplit(self):
		if len(self.next) > 1:
			return True
		else:
			return False

	def isUnion(self):
		if len(self.prev) > 1:
			return True
		else:
			return False


