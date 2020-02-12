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
		print("\n\n\n")
		print(model)
		#print(model.input_spec)
		self.buildMILRModel()



#Current Issue, names are for inputs not outputs....
#Solution Build backwards....

	def buildMILRModel(self):
		self.config = self.model.get_config()['layers']
		print(self.config)
		self.waitQ = []

		if type(self.model.layers[0]) != L.InputLayer:
			print(type(self.model.layers[0]))
			print("First Layer not Input Layer")
			sys.exit()

		self.milr = M.inputLayer(self.model.layers[0])
		self.tail = self.makeLayer(self.model.layers[-1], None)
		self.tail.isEnd()

		print("_____Builder______")
		self.builder(self.tail)
		print("__________________")
		self.print()


		del self.config
		del self.waitQ
		

	def builder(self, tail):
		#print(self.model.get_layer(name=head.name))
		#nextName = self.config.fromkeys()

		nextName = [i for i in self.config if i['name'] == tail.name][0]['inbound_nodes'][0]
		if len(nextName) > 1:
			print("Handling a Split")
			print("Single Path Handling")
		nextName = nextName[0][0]

		print(reversed(self.model.layers[:-1]))

		for layer in reversed(self.model.layers[:-1]):
			if layer.name != nextName:
				continue

			if layer == self.milr.Tlayer:
				self.milr.setNext(tail)
				break

			new = self.makeLayer(layer, tail)
			tail.setPrev(new)
			tail = new

			nextName = [i for i in self.config if i['name'] == tail.name][0]['inbound_nodes'][0]

			if len(nextName) > 1:
				print("Handling a Split")
				print("Single Path Handling")

			nextName = nextName[0][0]

			
			

	def makeLayer(self, layers, next):
		t = type(layers)

		if t == L.Flatten:
			return M.flattenLayer(layers, next = next)

		elif t == L.Dense:
			return M.denseLayer(layers, next = next)

		elif t ==  L.InputLayer:
			self.milr.setNext(next)
			return self.milr

		else:
			print(t)
			print("ERROR:Missing a Layer Type")
			return M.layerNode(layers, next = next)

	def initalize(self):
		return

		#milr = self.milrModel
		#inputShape = milr.getShape()
		#print(inputShape)
	

	#simplified as it doesnt account for splits	
	def print(self):
		head = self.milr

		while True:
			print(head)
			if head.hasNext():
				head = head.move()
			else :
				break


	def test(self):
		model = self.model

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

