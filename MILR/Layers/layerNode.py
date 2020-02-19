class layerNode:

	def __init__(self, layer, prev = None, next = None, end = False):
		self.prev = [prev]
		self.next = [next]
		self.end = end
		self.Tlayer = layer
		print(layer.name)
		self.name = layer.name
		self.inputLayer = False
		self.inputSize = None
		self.outputSize = None

	def __str__(self):
		return self.name 
		#+ " Next " + str(len(self.next)) + " Prev " + str(len(self.prev))

	def setAsInputLayer(self):
		self.inputLayer = True
		self.inputSize = self.Tlayer.get_config()['batch_input_shape'][1:]

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


