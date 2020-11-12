import tensorflow as tf
from tensorflow import keras
import numpy as np
from Crypto.Cipher import AES
import sys

#Handles the Simplifivation of TF Layer Classing
from tensorflow.keras import layers as L

# Handles the MILR Layers Classes
import MILR
from MILR.status import status as STAT
from tensorflow.python.keras.layers.normalization import BatchNormalization

# error Sim
from random import random, seed
import struct
import time
import os

#Raw bit Error Rate (RBER) each bit in the binary array will be flipped independently with some probability p 
def continousRecoveryTest(NET,rounds, error_Rate, testFunc, TestingData, testNumber):
	if not os.path.exists('data'):
		os.makedirs('data')
	print("data/{}-continousRecoveryTest.csv".format(testNumber))
	fout = open(("data/{}-continousRecoveryTest.csv".format(testNumber)), "w")

	seed()
	baslineAcc = testFunc(*TestingData)

	rawWeights = NET.model.get_weights()

	for rates in error_Rate:
		NET.model.set_weights(rawWeights)
		beginRoundACC = baslineAcc
		for z in range(1,rounds+1):
			print("\nBegin round {}, errorRate {}".format(z,rates))
			errorCount = 0
			errorLayers = []
			for l in range(len(NET.milrModel)):
				layer = NET.milrModel[l]
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
							error, sets[i] = floatError(rates, sets[i])
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

			start_time = time.time()
			error, doubleError,kernBiasError,TIME, log = NET.scrubbing(retLog = True)
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

			scrubAcc = testFunc(*TestingData)

			print("{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , beginRoundACC, errorCount, len(errorLayers), errAcc, len(log), logAcc, scrubAcc, tTime, doubleError, kernBiasError))
			fout.write("{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , beginRoundACC, errorCount, len(errorLayers), errAcc, len(log), logAcc, scrubAcc, tTime, doubleError,kernBiasError))
			beginRoundACC = scrubAcc

	fout.close()

def eccMILR(NET,rounds, error_Rate, testFunc, TestingData, testNumber):
	if not os.path.exists('data3'):
		os.makedirs('data3')
	print("data3/{}-eccMILR.csv".format(testNumber))
	fout = open("data3/{}-eccMILR.csv".format(testNumber), "w")

	rawWeights = NET.model.get_weights()

	seed()
	baslineAcc = testFunc(*TestingData)

	for rates in error_Rate:
		for z in range(1,rounds+1):
			print("\nBegin round {}, errorRate {}".format(z,rates))
			doubleErrorFlag = False
			kernBiasError = False
			NET.model.set_weights(rawWeights)
			errorCount = 0
			errorLayers = []
			errLay = []
			errorInCheck = False
			for l in range(len(NET.milrModel)):
				layer = NET.milrModel[l]
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
							error, sets[i], count = floatErrorECC(rates, sets[i])
							if error:
								errorCount += count
								layerErrorCount+=1
								errorOnThisLayer = True
								subLayerErr = True
						sets = np.reshape(sets, shape)
						weights[j] = sets
						if subLayerErr:
							if localDoubelError:
								kernBiasError = True
							localDoubelError = True
				if errorOnThisLayer:
					if errorInCheck:
						doubleErrorFlag = True
					errorInCheck = True
				layer.setWeights(weights)
				if errorOnThisLayer:
					errorLayers.append((layer.name,layerErrorCount))
					errLay.append(l)
				#print(layer, layerErrorCount)
			#print(errorCount)

			errAcc = testFunc(*TestingData)

			error, doubleError,kernBiasError,TIME, log = NET.scrubbing(retLog = True)
			scrubAcc = testFunc(*TestingData)

			if len(log) != len(errLay):
				logAcc = False
			else:
				for l1, l2 in zip(log, errLay):
					if l1[1] != l2:
						logAcc = False
						break
				logAcc = True

			if not logAcc:
				print(log)

			TIME = str(TIME[0]) + ";" + str(TIME[1])

			fout.write("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
			print("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
	fout.close()

#Raw bit Error Rate (RBER) each bit in the binary array will be flipped independently with some probability p 
def RBERefftec(NET,rounds, error_Rate, testFunc, TestingData, testNumber):
	if not os.path.exists('data3'):
		os.makedirs('data3')
	print("data3/{}-RBEREffect.csv".format(testNumber))
	fout = open("data3/{}-RBEREffect.csv".format(testNumber), "w")

	rawWeights = NET.model.get_weights()

	seed()
	baslineAcc = testFunc(*TestingData)

	for rates in error_Rate:
		for z in range(1,rounds+1):
			print("\nBegin round {}, errorRate {}".format(z,rates))
			doubleErrorFlag = False
			kernBiasError = False
			NET.model.set_weights(rawWeights)
			errorCount = 0
			errorLayers = []
			errLay = []
			errorInCheck = False
			for l in range(len(NET.milrModel)):
				layer = NET.milrModel[l]
				if layer.checkpointed:
					errorInCheck = False
				errorOnThisLayer = False
				layerErrorCount = 0
				weights = layer.getWeights()
				if weights is not None:
					layer.biasError = False
					localDoubelError = False
					for j in range(len(weights)):
						subLayerErr = False
						sets = np.array(weights[j])
						shape = sets.shape
						sets  = sets.flatten()
						for i in range(len(sets)):
							#error, sets[i], count = floatErrorWhole(rates, sets[i])
							error, sets[i], count = floatError(rates, sets[i])
							if error:
								errorCount += count
								layerErrorCount+=1
								errorOnThisLayer = True
								subLayerErr = True
						sets = np.reshape(sets, shape)
						weights[j] = sets
						if subLayerErr:
							if l == 1:
								layer.biasError = True
							if localDoubelError:
								kernBiasError = True
								layer.biasError = False
							localDoubelError = True
				if errorOnThisLayer:
					if errorInCheck:
						doubleErrorFlag = True
					errorInCheck = True
				layer.setWeights(weights)
				if errorOnThisLayer:
					errorLayers.append((layer.name,layerErrorCount))
					errLay.append(l)
				#print(layer, layerErrorCount)
			#print(errorCount)

			errAcc = testFunc(*TestingData)

			# errorWeights = NET.model.get_weights()

			error, doubleError,kernBiasError, TIME, log = NET.scrubbing(retLog = True)
			scrubAcc = testFunc(*TestingData)
			# NET.model.set_weights(errorWeights)

			# locallog = errorIdentFromErrorList(errLay)
			# NET.recovery(locallog)
			# perAcc = testFunc(*TestingData)
			# print(locallog)

			if len(log) != len(errLay):
				logAcc = False
			else:
				for l1, l2 in zip(log, errLay):
					if l1[1] != l2:
						logAcc = False
						break
				logAcc = True

			TIME = str(TIME[0]) + ";" + str(TIME[1])

			fout.write("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
			print("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
	fout.close()

def RBERefftecWhole(NET,rounds, error_Rate, testFunc, TestingData, testNumber):
	if not os.path.exists('data3'):
		os.makedirs('data3')
	print("data3/{}-RBEREffectWhole.csv".format(testNumber))
	fout = open("data3/{}-RBEREffectWhole.csv".format(testNumber), "w")

	rawWeights = NET.model.get_weights()

	seed()
	baslineAcc = testFunc(*TestingData)

	for rates in error_Rate:
		for z in range(1,rounds+1):
			print("\nBegin round {}, errorRate {}".format(z,rates))
			doubleErrorFlag = False
			kernBiasError = False
			NET.model.set_weights(rawWeights)
			errorCount = 0
			errorLayers = []
			errLay = []
			errorInCheck = False
			for l in range(len(NET.milrModel)):
				layer = NET.milrModel[l]
				if layer.checkpointed:
					errorInCheck = False
				errorOnThisLayer = False
				layerErrorCount = 0
				weights = layer.getWeights()
				if weights is not None:
					layer.biasError = False
					localDoubelError = False
					for j in range(len(weights)):
						subLayerErr = False
						sets = np.array(weights[j])
						shape = sets.shape
						sets  = sets.flatten()
						for i in range(len(sets)):
							error, sets[i], count = floatErrorWhole(rates, sets[i])
							if error:
								errorCount += count
								layerErrorCount+=1
								errorOnThisLayer = True
								subLayerErr = True
						sets = np.reshape(sets, shape)
						weights[j] = sets
						if subLayerErr:
							if l == 1:
								layer.biasError = True
							if localDoubelError:
								kernBiasError = True
								layer.biasError = False
							localDoubelError = True
				if errorOnThisLayer:
					if errorInCheck:
						doubleErrorFlag = True
					errorInCheck = True
				layer.setWeights(weights)
				if errorOnThisLayer:
					errorLayers.append((layer.name,layerErrorCount))
					errLay.append(l)
				#print(layer, layerErrorCount)
			#print(errorCount)

			errAcc = testFunc(*TestingData)

			# errorWeights = NET.model.get_weights()

			error, doubleError,kernBiasError, TIME, log = NET.scrubbing(retLog = True)
			scrubAcc = testFunc(*TestingData)
			# NET.model.set_weights(errorWeights)

			# locallog = errorIdentFromErrorList(errLay)
			# NET.recovery(locallog)
			# perAcc = testFunc(*TestingData)
			# print(locallog)

			if len(log) != len(errLay):
				logAcc = False
			else:
				for l1, l2 in zip(log, errLay):
					if l1[1] != l2:
						logAcc = False
						break
				logAcc = True

			TIME = str(TIME[0]) + ";" + str(TIME[1])

			fout.write("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
			print("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
	fout.close()

def LayerSpecefic(NET,rounds, error_Rate, testFunc, TestingData, testNumber):
	if not os.path.exists('data3'):
		os.makedirs('data3')
	print("data3/{}-LayerSpecefic.csv".format(testNumber))
	fout = open("data3/{}-LayerSpecefic.csv".format(testNumber), "w")

	rawWeights = NET.model.get_weights()

	seed()
	baslineAcc = testFunc(*TestingData)

	for rates in error_Rate:
		for l in range(len(NET.milrModel)):
			layer = NET.milrModel[l]
			weights = layer.getWeights()
			if weights is not None:
				for z in range(1,rounds+1):
					errorCount = 0
					weights = layer.getWeights()
					for j in range(len(weights)):
						TIME = (0,0)
						sets = np.array(weights[j])
						shape = sets.shape
						sets = np.random.rand(*shape)
						weights[j] = sets
						layer.setWeights(weights)
				
						errAcc = testFunc(*TestingData)
						scrubAcc = 0
						if not (j==0 and type(layer) == M.convolutionLayer2d and layer.CRC == True):
							error, doubleError,kernBiasError, TIME, log = NET.scrubbing(retLog = True)
							scrubAcc = testFunc(*TestingData)

						TIME = str(TIME[0]) + ";" + str(TIME[1])

						fout.write("{};{};{};{};{};{};{};{};{}\n".format(rates, z , layer,j,baslineAcc, errorCount, errAcc,scrubAcc, TIME))
						print("{};{};{};{};{};{};{};{};{}\n".format(rates, z , layer,j, baslineAcc, errorCount, errAcc,scrubAcc, TIME))
						NET.model.set_weights(rawWeights)
						weights = layer.getWeights()
	fout.close()

#Raw bit Error Rate (RBER) each bit in the binary array will be flipped independently with some probability p 
def AESErrors(NET,rounds, error_Rate, testFunc, TestingData, testNumber):
	if not os.path.exists('data3'):
		os.makedirs('data3')
	print("data3/{}-AESErrors.csv".format(testNumber))
	fout = open("data3/{}-AESErrors.csv".format(testNumber), "w")

	rawWeights = NET.model.get_weights()

	seed()
	baslineAcc = testFunc(*TestingData)

	for rates in error_Rate:
		for z in range(1,rounds+1):
			print("\nBegin round {}, errorRate {}".format(z,rates))
			doubleErrorFlag = False
			kernBiasError = False
			NET.model.set_weights(rawWeights)
			errorCount = 0
			errorLayers = []
			errLay = []
			errorInCheck = False
			for l in range(len(NET.milrModel)):
				layer = NET.milrModel[l]
				if layer.checkpointed:
					errorInCheck = False
				errorOnThisLayer = False
				layerErrorCount = 0
				weights = layer.getWeights()
				if weights is not None:
					layer.biasError = False
					localDoubelError = False
					for j in range(len(weights)):
						subLayerErr = False
						sets = np.array(weights[j])
						shape = sets.shape
						sets  = sets.flatten()
						for i in range(int(len(sets)/4)):
							# pass lengths of 4
							error, val, count = AES_Insert_Errors(rates, sets[i*4:(i*4)+4])

							for z in range(4):
								sets[(i*4)+z] = val[z]

							if error:
								errorCount += count
								layerErrorCount+=1
								errorOnThisLayer = True
								subLayerErr = True
						sets = np.reshape(sets, shape)
						weights[j] = sets
						if subLayerErr:
							if l == 1:
								layer.biasError = True
							if localDoubelError:
								kernBiasError = True
								layer.biasError = False
							localDoubelError = True
				if errorOnThisLayer:
					if errorInCheck:
						doubleErrorFlag = True
					errorInCheck = True
				layer.setWeights(weights)
				if errorOnThisLayer:
					errorLayers.append((layer.name,layerErrorCount))
					errLay.append(l)
				#print(layer, layerErrorCount)
			#print(errorCount)

			errAcc = testFunc(*TestingData)

			# errorWeights = NET.model.get_weights()

			error, doubleError,kernBiasError, TIME, log = NET.scrubbing(retLog = True)
			scrubAcc = testFunc(*TestingData)
			# NET.model.set_weights(errorWeights)

			# locallog = errorIdentFromErrorList(errLay)
			# NET.recovery(locallog)
			# perAcc = testFunc(*TestingData)
			# print(locallog)

			if len(log) != len(errLay):
				logAcc = False
			else:
				for l1, l2 in zip(log, errLay):
					if l1[1] != l2:
						logAcc = False
						break
				logAcc = True

			TIME = str(TIME[0]) + ";" + str(TIME[1])

			fout.write("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
			print("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
	fout.close()


#Raw bit Error Rate (RBER) each bit in the binary array will be flipped independently with some probability p 
def AES_ECC_Errors(NET,rounds, error_Rate, testFunc, TestingData, testNumber):
	if not os.path.exists('data3'):
		os.makedirs('data3')
	print("data3/{}-AESeccErrors.csv".format(testNumber))
	fout = open("data3/{}-AESeccErrors.csv".format(testNumber), "w")

	rawWeights = NET.model.get_weights()

	seed()
	baslineAcc = testFunc(*TestingData)

	for rates in error_Rate:
		for z in range(1,rounds+1):
			print("\nBegin round {}, errorRate {}".format(z,rates))
			doubleErrorFlag = False
			kernBiasError = False
			NET.model.set_weights(rawWeights)
			errorCount = 0
			errorLayers = []
			errLay = []
			errorInCheck = False
			for l in range(len(NET.milrModel)):
				layer = NET.milrModel[l]
				if layer.checkpointed:
					errorInCheck = False
				errorOnThisLayer = False
				layerErrorCount = 0
				weights = layer.getWeights()
				if weights is not None:
					layer.biasError = False
					localDoubelError = False
					for j in range(len(weights)):
						subLayerErr = False
						sets = np.array(weights[j])
						shape = sets.shape
						sets  = sets.flatten()
						for i in range(int(len(sets)/4)):
							# pass lengths of 4
							error, val, count = AESecc_Insert_Errors(rates, sets[i*4:(i*4)+4])

							for z in range(4):
								sets[(i*4)+z] = val[z]

							if error:
								errorCount += count
								layerErrorCount+=1
								errorOnThisLayer = True
								subLayerErr = True
						sets = np.reshape(sets, shape)
						weights[j] = sets
						if subLayerErr:
							if l == 1:
								layer.biasError = True
							if localDoubelError:
								kernBiasError = True
								layer.biasError = False
							localDoubelError = True
				if errorOnThisLayer:
					if errorInCheck:
						doubleErrorFlag = True
					errorInCheck = True
				layer.setWeights(weights)
				if errorOnThisLayer:
					errorLayers.append((layer.name,layerErrorCount))
					errLay.append(l)
				#print(layer, layerErrorCount)
			#print(errorCount)

			errAcc = testFunc(*TestingData)

			# errorWeights = NET.model.get_weights()

			error, doubleError,kernBiasError, TIME, log = NET.scrubbing(retLog = True)
			scrubAcc = testFunc(*TestingData)
			# NET.model.set_weights(errorWeights)

			# locallog = errorIdentFromErrorList(errLay)
			# NET.recovery(locallog)
			# perAcc = testFunc(*TestingData)
			# print(locallog)

			if len(log) != len(errLay):
				logAcc = False
			else:
				for l1, l2 in zip(log, errLay):
					if l1[1] != l2:
						logAcc = False
						break
				logAcc = True

			TIME = str(TIME[0]) + ";" + str(TIME[1])

			fout.write("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
			print("{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(rates, z , baslineAcc, errorCount, len(errorLayers), errAcc, errorLayers, TIME, doubleErrorFlag, kernBiasError, scrubAcc, logAcc, len(log)))
	fout.close()

def errorIdentFromErrorList(NET, list):
	errorFlag = False
	errorFlagLoc = 0
	erroLog = []
	checkMark = 0

	listIndex = 0

	for i in range(len(NET.milrModel)):
		if i == list[listIndex]:
			listIndex +=1
			error = True
		else:
			error = False

		if NET.milrModel[i].checkpointed:
			if errorFlag:
				erroLog.append((checkMark,errorFlagLoc, i))
			checkMark = i
			errorFlag = False

		if error:
			print("error:", NET.milrModel[i])
			if errorFlag:
				print("Two Errors between checkpoints")
			errorFlag = True
			errorFlagLoc = i

	if errorFlag:
		erroLog.append((checkMark,errorFlagLoc, -1))

	return erroLog

def AESecc_Insert_Errors(error_Rate, num):
	Key = os.urandom(16)
	AES_ECB = AES.new(Key, AES.MODE_ECB)

	rawval = num.tobytes()
	encrypt = AES_ECB.encrypt(rawval)
	errorState, error, count = createErrors_ECC(error_Rate)

	errored = byte_xor(error, encrypt)

	decrypt = AES_ECB.decrypt(errored)

	return errorState, np.frombuffer(decrypt, dtype=num.dtype), count

def createErrors_ECC(error_Rate):
	main_error = bytearray(0)
	main_count = 0
	for k in range(4):
		error = bytearray(4)
		error = int.from_bytes(error, byteorder='big')
		count = 0

		if random() < error_Rate:
			error = error + 1
			count +=1

		for j in range(32):
			error = error << 1
			if random() < error_Rate:
				error = error + 1
				count +=1

		error = error.to_bytes(16, byteorder='big')

		if count == 1:
			count = 0
			error = bytearray(4)

		main_error = main_error + error
		main_count+= count

	return (main_count > 0), main_error, main_count

def floatErrorECC(error_Rate, num):
	error = int(0)
	count = 0
	if random() < error_Rate:
		error = error + 1
		count +=1
	for i in range(30):
		error = error << 1
		if random() < error_Rate:
			error = error + 1
			count +=1

	if count > 1:
		num = floatToBits(num)
		#print(num, bin(error))
		num = num ^ error
		#print(num)
		return True, bitsToFloat(num), count
	else:
		return False, num, 0

def AES_Insert_Errors(error_Rate, num):
	Key = os.urandom(16)
	AES_ECB = AES.new(Key, AES.MODE_ECB)

	rawval = num.tobytes()
	encrypt = AES_ECB.encrypt(rawval)
	errorState, error, count = createErrors(error_Rate)

	errored = byte_xor(error, encrypt)

	decrypt = AES_ECB.decrypt(errored)

	return errorState, np.frombuffer(decrypt, dtype=num.dtype), count


def byte_xor(ba1, ba2):
    return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])

def createErrors(error_Rate):
	error = bytearray(16)
	error = int.from_bytes(error, byteorder='big')
	count = 0

	if random() < error_Rate:
		error = error + 1
		count +=1

	for j in range(127):
		error = error << 1
		if random() < error_Rate:
			error = error + 1
			count +=1

	error = error.to_bytes(16, byteorder='big')
	return (count > 0), error, count

def floatError(error_Rate, num):
	error = int(0)
	count = 0
	if random() < error_Rate:
		error = error + 1
		count +=1
	#63 for 64 bits
	for i in range(30):
		error = error << 1
		if random() < error_Rate:
			error = error + 1
			count +=1
	
	if count > 0:
		num = floatToBits(num)
		#print(num, bin(error))
		num = num ^ error
		#print(num)
		return True, bitsToFloat(num), count
	else:
		return False, num, 0

def floatErrorWhole(error_Rate, num):
	if random() < error_Rate:
		num = floatToBits(num)
		num = ~num
		return True,bitsToFloat(num), 1
	else:
		return False, num, 0

def floatToBits(f):
	#print(type(f))
	#64 bit d
	s = struct.pack('>f', f)
	#print(s)
	#64 bit Q
	return struct.unpack('>l', s)[0]

def bitsToFloat(b):
	#print(hex(b))
	s = struct.pack('>l', b)
	#print(type(s))
	return struct.unpack('>f', s)[0]