class layerNode:

	def __init__(self, layer, prev=None, next = None, end = False, secondIn = False, secondOut = False):
		self.prev = prev
		self.next = next
		self.end = end
		self.secondIn = secondIn
		self.secondOut = secondOut
		self.Tlayer = layer
		print(layer.name)
		self.name = layer.name


	def setNext(self, next):
		self.next = next

	def setPrev(self, prev):
		self.prev = prev

	def nextAndMove(self,node):
		self.next = node
		return node, self

	def hasNext(self):
		return not (self.end)

	def isEnd(self):
		self.end = True

	def move(self):
		return self.next



