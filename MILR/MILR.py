import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

#Handles the Simplifivation of TF Layer Classing
from tensorflow.keras import layers as L

# Handles the MILR Layers Classes
import MILR.Layers as M
from MILR.status import status as STAT
from tensorflow.python.keras.layers.normalization import BatchNormalization

# error Sim
from random import random, seed
import struct
import time
import os

class MILR:

	# model -> original model of network
	# milrModel -> array of milr layers that have direct corelation to model layer order
	# milrHead -> first layer of layer linked list model, same layers as in the milrModel

	def __init__(self, model):
		self.model = model
		self.milrModel = [None for L in model.layers]
		print("\n")

		self.buildMILRModel()

		for layer in self.milrModel:
			assert layer != None, "Error: We have a None Layer"

		#self.print(self.milrHead)
		#self.reverseprint(self.milrModel[-1])
		#self.splitprint(self.milrHead)
		self.model
		self.initalize()
		self.totalCost(printTotals=True, printLayers=True)
		#self.sequential = True

	def initalize(self):
		self.milrHead.initilize()

	#Raw bit Error Rate (RBER) each bit in the binary array will be flipped independently with some probability p 
	def continousRecoveryTest(self,rounds, error_Rate, testFunc, TestingData, testNumber):
		if not os.path.exists('data'):
			os.makedirs('data')
		print("data/{}-continousRecoveryTest.csv".format(testNumber))
		fout = open(("data/{}-continousRecoveryTest.csv".format(testNumber)), "w")
		#fout.write()
	
		seed()
		baslineAcc = testFunc(*TestingData)
		print("Basline Accuracy:", baslineAcc)
		beginRoundACC = baslineAcc

		for z in range(1,rounds+1):
			errorCount = 0
			errorLayers = []
			for l in range(len(self.milrModel)):
				layer = self.milrModel[l]
				errorOnThisLayer = False
				layerErrorCount = 0
				weights = layer.getWeights()
				if weights is not None:
					#print("pre",weights)
					for j in range(len(weights)):
						sets = np.array(weights[j])
						setspre = sets[:]
						shape = sets.shape
						sets  = sets.flatten()
						for i in range(len(sets)):
							error, sets[i] = self.floatError(error_Rate, sets[i])
							if error:
								errorCount += 1
								layerErrorCount+=1
								if not errorOnThisLayer:
									errorLayers.append(l)
									errorOnThisLayer = True
						sets = np.reshape(sets, shape)
						weights[j] = sets
				layer.setWeights(weights)
				print(layer, layerErrorCount)
			print(errorCount)

			errAcc = testFunc(*TestingData)
			print("Pre Scrubbing Round {} , Acurracyr".format(z),errAcc)

			start_time = time.time()
			error, doubleError, log = self.scrubbing(retLog = True)
			end_time = time.time()
			tTime =  end_time - start_time
			print("Time: ", tTime)



			if len(log) != len(errorLayers):
				logAcc = False
			else:
				for l1, l2 in zip(log, errorLayers):
					if l1[1] != l2:
						logAcc = False
						break
				logAcc = True

			if error:
				print("Errors in round: ", z)

			scrubAcc = testFunc(*TestingData)
			print("Round {} , Acurracyr".format(z),scrubAcc)

			print("{},{},{},{},{},{},{},{},{},{},{}, {}\n".format(error_Rate, testNumber, z , beginRoundACC, errorCount, len(errorLayers), errAcc, len(log), logAcc, scrubAcc, tTime, doubleError))
			fout.write("{},{},{},{},{},{},{},{},{},{},{}, {}\n".format(error_Rate, testNumber, z , beginRoundACC, errorCount, len(errorLayers), errAcc, len(log), logAcc, scrubAcc, tTime, doubleError))
			beginRoundACC = scrubAcc

		fout.close()

	#Raw bit Error Rate (RBER) each bit in the binary array will be flipped independently with some probability p 
	def RBERefftec(self,rounds, error_Rate, testFunc, TestingData):
		if not os.path.exists('data'):
			os.makedirs('data')
		print("data/RBEREffect.csv")
		fout = open(("data/RBEREffect.csv"), "w")
		#fout.write()

		rawWeights = self.model.get_weights()
	
		seed()
		baslineAcc = testFunc(*TestingData)
		print("Basline Accuracy:", baslineAcc)

		for rates in error_Rate:
			for z in range(1,rounds+1):
				doubleErrorFlag = False
				self.model.set_weights(rawWeights)
				errorCount = 0
				errorLayers = []
				errorInCheck = False
				for l in range(len(self.milrModel)):
					layer = self.milrModel[l]
					if layer.checkpointed:
						errorInCheck = False
					errorOnThisLayer = False
					layerErrorCount = 0
					weights = layer.getWeights()
					if weights is not None:
						localDoubelError = False
						for j in range(len(weights)):
							subLayerErr = False
							sets = np.array(weights[j])
							shape = sets.shape
							sets  = sets.flatten()
							for i in range(len(sets)):
								error, sets[i] = self.floatError(rates, sets[i])
								if error:
									errorCount += 1
									layerErrorCount+=1
									errorOnThisLayer = True
									subLayerErr = True
							sets = np.reshape(sets, shape)
							weights[j] = sets
							if subLayerErr:
								if localDoubelError:
									doubleErrorFlag = True
								localDoubelError = True
					if errorOnThisLayer:
						if errorInCheck:
							doubleErrorFlag = True
						errorInCheck = True
					layer.setWeights(weights)
					if errorOnThisLayer:
						errorLayers.append((layer.name,layerErrorCount))
					print(layer, layerErrorCount)
				print(errorCount)

				errAcc = testFunc(*TestingData)
				print("Error ACC Round {} , Acurracyr".format(z),errAcc)

				fout.write("{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, doubleErrorFlag))
				print("{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, doubleErrorFlag))
		fout.close()
	"""
	def nonSeqScrubbing(self, retLog=False):
		erroLog = []

		self.milrHead.nonSeqScrubbing(erroLog)
	"""

	def errorIdent(self):
		doubleErrorFlag = False
		errorFlag = False
		errorFlagLoc = 0
		erroLog = []
		checkMark = 0

		for i in range(len(self.milrModel)):
			check, error, dbError = self.milrModel[i].partialCheckpoint()

			if dbError:
				print("Doubel Error Bias/Kern")
				doubleErrorFlag = True

			if check:
				if errorFlag:
					erroLog.append((checkMark,errorFlagLoc, i))
				checkMark = i
				errorFlag = False

			if error:
				print("error:", self.milrModel[i])
				if errorFlag:
					print("Two Errors between checkpoints")
					doubleErrorFlag = True
				errorFlag = True
				errorFlagLoc = i

		if errorFlag:
			erroLog.append((checkMark,errorFlagLoc, -1))
		#erroLog.append((checkMark,len(self.milrModel)-1, -1))

		return erroLog, doubleErrorFlag

	def scrubbing(self, retLog = False):
		print("Start scrubbing")

		#if not self.sequential:
			#return self.nonSeqScrubbing(retLog = retLog)

		erroLog, doubleErrorFlag = self.errorIdent()
		
		# Error Solving
		
		for log in erroLog:

			inputs = self.milrModel[log[0]].getCheckpoint()
			for i in range(log[0],log[1]):
				#begin to error
				inputs = self.milrModel[i].forwardPass(inputs)

			if log[2] == -1:
				outputs = self.milrModel[-1].outputData
				for i in range(len(self.milrModel)-1, log[1], -1):
					outputs = self.milrModel[i].backwardPass(outputs)
			else:
				outputs = self.milrModel[log[2]].getCheckpoint()
				for i in range(log[2]-1, log[1], -1):
					outputs = self.milrModel[i].backwardPass(outputs)

			print("Recovered ", self.milrModel[log[1]])
			self.milrModel[log[1]].kernelSolver(inputs, outputs)

		print("scrubbing complete")

		if retLog:
			return len(erroLog) > 0,doubleErrorFlag,  erroLog

		return len(erroLog) > 0, doubleErrorFlag

	def totalCost(self, printTotals=False, printLayers=False):
		checkSum = 0
		partSum = 0
		storedSum = 0

		for layer in self.milrModel:
			check, part, stored = layer.cost()

			if printLayers:
				print(layer)
				print("	Checkpoint Cost: ", check)
				print("	Partial Checkpoint Cost: ", part)
				print("	Stored Additional Data: ", stored)
				print("	Total Cost: ", check + part + stored)

			checkSum += check
			partSum += part
			storedSum += stored
		total = checkSum + partSum + storedSum

		if printTotals:
			print("Checkpoint Cost: ", checkSum)
			print("Partial Checkpoint Cost: ", partSum)
			print("Stored Additional Data: ", storedSum)
			print("Total Cost: ", total)
		return total


	def floatError(self, error_Rate, num):
		error = int(0)
		for i in range(31):
			if random() < error_Rate:
				error = error + 1
			error = error << 1

		
		if error > 0:
			num = self.floatToBits(num)
			#print(num, bin(error))
			num = num ^ error
			#print(num)
			return error > 0, self.bitsToFloat(num)
		else:
			return False, num


	def floatToBits(self, f):
		s = struct.pack('>f', f)
		return struct.unpack('>I', s)[0]

	def bitsToFloat(self, b):
		s = struct.pack('>I', b)
		return struct.unpack('>f', s)[0]


	def buildMILRModel(self):
		self.config = self.model.get_config()['layers']
		self.milrHead = self.makeLayer(self.model.layers[0], None)
		self.milrHead.setAsInputLayer()
		self.milrModel[0] = self.milrHead
		tail = self.makeLayer(self.model.layers[-1], None)
		self.milrModel[-1] = tail
		self.builder(tail)
		tail.isEnd()
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

		if t == BatchNormalization:
			return M.batchNormalization(layers, next = next)

		elif t == L.Conv2D:
			return M.convolutionLayer2d(layers,next = next)

		elif t == L.Dense:
			return M.denseLayer(layers, next = next)

		elif t == L.Activation:
			return M.activationLayer(layers,next = next)

		elif t == L.Add:
			#self.sequential = False
			return M.addLayer(layers,next = next)

		elif t == L.ZeroPadding2D:
			return M.zeroPaddingLayer(layers,next = next)

		# Non Invertible
		# No Weights
		# Checkpoint when encountered
		elif t == L.MaxPooling2D or t == L.GlobalAveragePooling2D:
			return M.NonInv_Check(layers,next = next)

		elif t == L.Flatten:
			return M.flattenLayer(layers, next = next)

		# Passthrough Layers,do no operations in inference
		# No change in shape
		# Invertible and No weight modification
		elif t == L.Dropout or t == L.InputLayer:
			return M.layerNode(layers, next = next)

		else:
			print(t)
			print("ERROR:Missing a Layer Type")
			return M.layerNode(layers, next = next)	

#Prints sinlge list with signalling of splits and unions
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
				print(head)
				break

#Prints sinlge list with signalling of splits and unions in Reverse
	def reverseprint(self, head):
		while True:
			if head.hasPrev():
				nexts = head.getPrev()

				if head.isSplit():
					print("		Union")
					return head

				if head.isUnion():
					print(head)
					print("		SPLIT")
					for n in nexts:
						head = self.reverseprint(n)
					print(head)
					head = head.getPrev()[0]
				else: 
					print(head)
					head = nexts[0]
			else :
				print(head)
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
			