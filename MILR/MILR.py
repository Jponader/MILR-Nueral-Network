import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys



#Handles the Simplifivation of TF Layer Classing
from tensorflow.keras import layers as L

# Handles the MILR Layers Classes
import MILR.Layers as M

class MILR:

	def __init__(self, model):
		self.model = model
		self.milrModel = [None for L in model.layers]
		print("\n\n\n")
		#print(model)
		#print(model.input_spec)
		#self.test()
		self.buildMILRModel()

		for layer in self.milrModel:
			if layer == None:
				print("Error: We have a None")



#Current Issue, names are for inputs not outputs....
#Solution Build backwards....

	def buildMILRModel(self):
		self.config = self.model.get_config()['layers']
		#print(self.config)

		self.milrHead = self.makeLayer(self.model.layers[0], None)
		self.milrHead.setAsInputLayer()
		self.milrModel[0] = self.milrHead
		tail = self.makeLayer(self.model.layers[-1], None)
		tail.isEnd()
		self.milrModel[-1] = tail

		print("_____Builder______")
		self.builder(tail)
		print("__________________")
		print(len(self.milrModel))
		print(len(self.model.layers))
		self.print(self.milrHead)
		self.splitprint(self.milrHead)
		
		del self.config
		

	def builder(self, tail):
		nextName = []
		layer = self.model.layers

		nameSet = [i for i in self.config if i['name'] == tail.name][0]['inbound_nodes'][0]
		for names in nameSet:
			nextName.append([names[0],tail])


		for pos in range(len(self.milrModel)-1, -1 , -1):
			print(pos)
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
					#print(nextName)
					del nextName[j]
					#print(nextName)

					nameSet = [i for i in self.config if i['name'] == new.name][0]['inbound_nodes'][0]
					for names in nameSet:
						nextName.append([names[0],new])

			

	def makeLayer(self, layers, next):
		t = type(layers)

		if t == L.Flatten:
			return M.flattenLayer(layers, next = next)

		elif t == L.Dense:
			return M.denseLayer(layers, next = next)

		elif t ==  L.InputLayer:
			return M.inputLayer(layers,next = next)

		elif t == L.ZeroPadding2D:
			return M.zeroPaddingLayer(layers,next = next)

		elif t == L.Conv2D:
			return M.convolutionLayer(layers,next = next)

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
			


	def test(self):
		model = self.model

		milrModel = [[model.layers],[None for L in model.layers]]
		array = model.layers
		print(milrModel)
		print(type(milrModel))


"""
		for layers in model.layers:
			print()

			print(type(layers))
			#Keras Layers Core
			print(layers.get_config())

			#Keras Engine Base_layer
			#print(layers.weights)

			#Module Module
			print(layers.name)
			#print(layers.name_scope)
			#print(layers.variables)
			#if len(layers.variables) > 0:
				#print(layers.variables[0])
				#print(type(layers.variables[0]))
			#print(layers.submodules)
"""
