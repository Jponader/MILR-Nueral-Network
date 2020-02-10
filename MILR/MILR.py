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
		#self.builder(self.milr)

		del self.config
		del self.waitQ
		

	def builder(self, head):
		#print(self.model.get_layer(name=head.name))
		#nextName = self.config.fromkeys()

		print(head.name)
		print(self.config[1]['name'])

		#while True:
		nextName = [i for i in self.config if i['name'] == head.name]
		print(nextName)

	def OldBuild(self):
		model = self.model


		if type(model.layers[0]) != L.InputLayer:
			print(type(model.layers[0]))
			print("First Layer not Input Layer")
			sys.exit()

		milr = M.inputLayer(model.layers[0])
		head = milr
		prev = None

		for layers in model.layers[1:]:
			t = type(layers)
			print(t)
			#print(layers.get_config())
			print(layers._inbound_nodes[0])

			if t == L.Flatten:
				head, prev = head.nextAndMove(M.flattenLayer(layers,prev = prev))

			elif t == L.Dense:
				head, prev = head.nextAndMove(M.denseLayer(layers,prev = prev))

			else:
				print(t)
				print("ERROR:Missing a Layer Type")

		head.isEnd()
		self.milrModel = milr

		#self.print()

		print(model.get_config())
		print(type(model.get_config()))

	def initalize(self):
		return

		#milr = self.milrModel
		#inputShape = milr.getShape()
		#print(inputShape)
	

	#simplified as it doesnt account for splits	
	def print(self):
		head = self.milrModel

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

