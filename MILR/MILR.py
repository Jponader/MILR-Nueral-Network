import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys



#Handles the Simplifivation of TF Layer Classing
from tensorflow.keras import layers as L

# Handles the MILR Layers Classes
import MILR.Layers as M
from tensorflow.python.keras.layers.normalization import BatchNormalization

class MILR:

	def __init__(self, model):
		self.model = model
		self.milrModel = [None for L in model.layers]
		print("\n")

		self.buildMILRModel()

		for layer in self.milrModel:
			if layer == None:
				print("Error: We have a None")





	def buildMILRModel(self):
		self.config = self.model.get_config()['layers']

		self.milrHead = self.makeLayer(self.model.layers[0], None)
		self.milrHead.setAsInputLayer()
		self.milrModel[0] = self.milrHead
		tail = self.makeLayer(self.model.layers[-1], None)
		tail.isEnd()
		self.milrModel[-1] = tail

		print("_____Builder______")
		print(tail)
		self.builder(tail)
		print(self.milrHead)
		print("__________________")
		#print(len(self.milrModel))
		#print(len(self.model.layers))
		#self.print(self.milrHead)
		self.splitprint(self.milrHead)
		del self.config
		

	def builder(self, tail):
		nextName = []
		layer = self.model.layers
		nameSet = [i for i in self.config if i['name'] == tail.name][0]['inbound_nodes'][0]
		for names in nameSet:
			nextName.append([names[0],tail])

		for pos in range(len(self.milrModel)-1, -1 , -1):
			for j in range(len(nextName)-1, -1,-1):
				if nextName[j][0] == layer[pos].name:
					print()
					if self.milrModel[pos] != None:
						new = self.milrModel[pos]
						new.setNext(nextName[j][1])

						if new == self.milrHead:
							nextName[j][1].setPrev(self.milrHead)
						else:
							nextName[j][1].setPrev(new)

						del nextName[j]
						break
					else:
						new = self.makeLayer(layer[pos], nextName[j][1])
						self.milrModel[pos] = new

					nextName[j][1].setPrev(new)
					del nextName[j]

					nameSet = [i for i in self.config if i['name'] == new.name][0]['inbound_nodes'][0]
					for names in nameSet:
						nextName.append([names[0],new])

			

	def makeLayer(self, layers, next):
		t = type(layers)

		#print(L)

		if t == BatchNormalization:
			return M.batchNormalization(layers, next = next)

		elif t == L.Conv2D:
			return M.convolutionLayer2d(layers,next = next)

		elif t == L.Dense:
			return M.denseLayer(layers, next = next)

		elif t == L.Activation:
			return M.activationLayer(layers,next = next)

		elif t == L.Add:
			return M.addLayer(layers,next = next)

		elif t == L.ZeroPadding2D:
			return M.zeroPaddingLayer(layers,next = next)

		elif t == L.MaxPooling2D:
			return M.poolingLayer2d(layers,next = next)

		elif t == L.GlobalAveragePooling2D:
			return M.globalPoolingLayer(layers,next = next)

		elif t == L.Flatten:
			return M.flattenLayer(layers, next = next)

		#Passthrough Layers,do no operations in inference mode
		elif t == L.Dropout or t == L.InputLayer:
			return M.layerNode(layers, next = next)

		else:
			print(t)
			print("ERROR:Missing a Layer Type")
			return M.layerNode(layers, next = next)

	def initalize(self):
		return

		#milr = self.milrModel
		#inputShape = milr.getShape()
		#print(inputShape)
	


	def print(self, head):
		while True:
			if head.hasNext():
				nexts = head.getNext()

				if head.isUnion():
					print("		Union")
					return head

				if head.isSplit():
					print(head)
					print("		SPLIT")
					for n in nexts:
						head = self.print(n)
					print(head)
					head = head.getNext()[0]
				else: 
					print(head)
					head = nexts[0]
			else :
				break

# Assumption only one split is going on at a time, a split is only two items
	def splitprint(self, head):
		left = head
		right = None

		while True:
			if right == None:
				print("	",left)
				if not left.hasNext():
					return
				next = left.getNext()
				if left.isSplit():
					if len(next) > 2:
						print("ERROR")
						sys.exit()
					right = next[1]
				left = next[0]

			elif left == right:
				print("	",left)
				right = None
				next = left.getNext()
				if left.isSplit():
					if len(next) > 2:
						print("ERROR")
						sys.exit()
					right = next[1]
				left = next[0]

			elif left.isUnion():
				print("	|	", right)
				if right.isSplit():
					print("ERROR")
					sys.exit()
				right = right.getNext()[0]

			elif right.isUnion():
				print(left, "	      |")
				if left.isSplit():
					print("ERROR")
					sys.exit()
				left = left.getNext()[0]

			else:
				print(left,"	", right)
				right = right.getNext()[0]
				left = left.getNext()[0]
				if left.isSplit() or right.isSplit():
					print("ERROR")
					sys.exit()
			