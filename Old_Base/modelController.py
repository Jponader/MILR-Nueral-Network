from denseLayer import denseLayer
from flattenLayer import flattenLayer
from biasLayer import biasLayer
from convolutionLayer import convolutionLayer
from activationLayer import activationLayer

from loss import loss
#from optomizer import optomizer

import math
import numpy as np
from random import seed
from random import randint
from random import random
from datetime import datetime
from itertools import zip_longest

class modelController:

	def __init__(self, model = []):
		self.model = model

	def add(self, layer):
		self.model.append(layer)

	def compile(self, inputShape, loss = None, optomizer = None):
		self.loss = loss
		self.optomizer = optomizer
		for layers in self.model:
			print(layers)
			inputShape = layers.compile(inputShape)


	def forwardPass(self, inputMat):
		out = inputMat

		for layers in self.model:
			out = layers.forwardPass(out)

		return out

	def trainForward(self, inputMat):
		store = [inputMat]

		for layers in self.model:
			store.append(layers.forwardPass(store[-1]))

		return store


	def batchWork(self, dataset):
		out=[]
		i = 0
		for item in dataset:
			out.append(self.forwardPass(item))
			print(i)
			i += 1

		out = np.array(out)
		return out

	def train(self, dataset, answerset, batch = 10, epochs = 1):
		
		for epoch in range(0,epochs):
			for item in range(0,len(dataset)):
				hold = self.trainForward(dataset[item])
				loss = self.loss(hold[-1], answerset[item])
				print(loss)

		return 0




#_______________________________________________________________________________________________
	

	
	def addCategories(self, mapping):
		#allows for mapping to catergories
		return 0

	def predict(self, inputMat):
		out = forwardPass(self, inputMat)
		#change so that they output is a prediction not just the result, correlate the result to something
		return out

	#Not tested, basically sudo code
	def accuracyTest(self, dataset, answerset):
		correct = 0

		for items, answer in datset, answerset:
			out = prediction(self, items)
			if out == answer:
				correct += 1

		return correct / datset.length



"""
	def milrIntilization(self,inputMat):
		#create array showing that each layer is untainted, true weight sizes
		self.tainted = []
		#creare array to store the extra information to recover
		self.metadata = []
		self.metadata.append(inputMat)

		out = inputMat
		#state will tell the user what information needs to be stored, and some information about the layers before or after it
		state = 1

		for layers in self.model:
			out, metadata, state = layers.milrIntilization(out, state)

			#apply some logic to know when and where to store metadata, and what that should be depending on sitaution
			self.metadata.append(metadata)

		self.metadata.append(out)
		return out


	def milrRecovery(self):
		#find tainted layer
		#work through metadata goign forwards and backwards from tainto to find start postions

		#for those ahaed for the point they use normal forward passs

		#for those behind the point they use back pass

		#once have input and output of layer along with metadata, recover the layer weights
		#reset the taint stat to untainted, restart the machine running


	#some form of error identification protocol randomly inserted to detect errors, 
	#maybe run through program ever so often to test state, while comparing out puts
	#once find error one has correct starting point and input already now to find proper output

"""