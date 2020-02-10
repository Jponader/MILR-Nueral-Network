import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import numpy as np
#from initializer import initializer as setup

class MILR:

	def __init__(self, model):
		self.model = model
		print(model)
		#print(model.input_spec)
		self.buildMILRModel()


	def buildMILRModel(self):
		model = self.model

		for layers in model.layers:
			if type(layers) == L.InputLayer:
				print(layers.name)
				string = str(type(layers))
				print(string)




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

