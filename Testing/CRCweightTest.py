import numpy as np
from zlib import crc32
import os
from math import ceil
from random import randint, seed
from random import random
import struct

CRCCODE = int('9d7f97d6',16)


def CRC2D(data):
	shape = data.shape
	output = [np.zeros((shape[0],int(ceil(shape[1]/4)))),np.zeros((int(ceil(shape[0]/4)),shape[1]))]

	for i in range(shape[0]):
		for j in range(int(ceil(shape[1]/4))):
			output[0][i][j] = crc32(data[i,j*4:(j*4)+4],CRCCODE)

	for i in range(int(ceil(shape[0]/4))):
		for j in range(shape[1]):
			output[1][i][j] = crc32(np.ascontiguousarray(data[i*4:(i*4)+4,j]),CRCCODE)	

	return output

def CRC2dErrorFinder(data1, data2):
	results = np.equal(data1[0], data2[0])
	results2 = np.equal(data1[1], data2[1])

	columns = np.argwhere(results == False)
	rows = np.argwhere(results2 == False)

	errorMatrix = []

	for col in columns:
		index = col[1] * 4
		for r in rows:
			if r[1] >= index and r[1]< index +4:
				check = r[0]*4
				if col[0] >= check and col[0]< check +4:
					errorMatrix.append([col[0],r[1]])

	return np.array(errorMatrix, dtype=np.int32)

def floatError(error_Rate, num):
	error = int(0)
	for i in range(31):
		if random() < error_Rate:
			error = error + 1
		error = error << 1

	
	if error > 0:
		num =floatToBits(num)
		#print(num, bin(error))
		num = num ^ error
		#print(num)
		return error > 0,bitsToFloat(num)
	else:
		return False, num


def floatToBits(f):
	s = struct.pack('>f', f)
	return struct.unpack('>I', s)[0]

def bitsToFloat(b):
	s = struct.pack('>I', b)
	return struct.unpack('>f', s)[0]



if not os.path.exists('data'):
			os.makedirs('data')

print("data/CRCweightErr0.csv")
fout = open("data/CRCweightErr0.csv", "w")


runs = 100000
numErrorRange = 16

for r in range(runs):
		flag = True
		matrix = np.random.rand(4,4)
		CRC = CRC2D(matrix)
		errArray = []

		numError = randint(0,15)

		for i in range(numError):
			x = randint(0,3)
			y = randint(0,3)

			for err in errArray:
				#print(err)
				if err == (x,y):
					flag = False

			if flag:
				errArray.append((x,y))
				matrix[x][y] = 0
				#random()


		errArray = np.array(errArray)


		CRCErr = CRC2D(matrix)
		errorLog = CRC2dErrorFinder(CRCErr, CRC)

		allfound = True

		for err in errArray:
			flag = True
			for inserts in errorLog:
				if err[0] == inserts[0] and err[1] == inserts[1]:
					flag = False
			if flag:
				allfound = False
				break

		fout.write("{};{};{};{};{}\n".format(r, len(errArray), len(errorLog), len(errorLog)-len(errArray), allfound))
		print("{};{};{};{};{}\n".format(r, len(errArray), len(errorLog), len(errorLog)-len(errArray), allfound))







