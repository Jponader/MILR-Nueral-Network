class layerNode:

	def __init__(self, prev=None, next = None, end = False, secondIn = False, secondOut = False):
		self.prev = prev
		self.next = next
		self.end = end
		self.secondIn = secondIn
		self.secondOut = secondOut



	def nextAndMove(self,node):
		self.next = node
		return node, self

	def hasNext(self):
		return not (self.end)

	def isEnd(self):
		self.end = True

	def move(self):
		return self.next



