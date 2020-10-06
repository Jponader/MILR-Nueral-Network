from Crypto.Cipher import AES
import random
import numpy as np
import struct
import time
import sys
import os


def AESTest(testNumber, iterations, errorCount, fout,fout_SIMP):

	for i in range(iterations):
	# Init Values for round
		Key = os.urandom(16)
		# print("Key:\t\t\t{0:0128b}".format(int(Key.hex(),16)))
		Tweak = os.urandom(16)
		# print("Tweak:\t\t\t{0:0128b}".format(int(Tweak.hex(),16)))
		PlainText_Original = os.urandom(16)
		# print("PlainText_Original:\t{0:0128b}".format(int(PlainText_Original.hex(),16)))

	# Init ECB mode AES, no change in state after each encryption 
		AES_ECB = AES.new(Key, AES.MODE_ECB)

	#AES XTS Encrypt
	# AES - XTS mode: 	|PLAIN| xor with tweak, encrypt, {xor with tweak} |CIPHER| {xor with tweak}, decrypt, xor with tweak |PLAIN|
		XTS_Tweaked = byte_xor(Tweak, PlainText_Original)
		# print("XTS_Tweaked:\t\t{0:0128b}".format(int(XTS_Tweaked.hex(),16)))
		XTS_Encrypt = AES_ECB.encrypt(XTS_Tweaked)
		# print("XTS_Encrypt:\t\t{0:0128b}".format(int(XTS_Encrypt.hex(),16)))
		XTS_Encrypt_Tweaked = byte_xor(Tweak, XTS_Encrypt)
		# print("XTS_Encrypt_Tweaked:\t{0:0128b}".format(int(XTS_Encrypt_Tweaked.hex(),16)))

	# AES Encrypt
	# AES Normal Mode:	|PLAIN|   encrypt  |CIPHER|  decrypt  |PLAIN|
		Plain_Encrypt  =AES_ECB.encrypt(PlainText_Original)		
		# print("Plain_Encrypt:\t\t{0:0128b}".format(int(Plain_Encrypt.hex(),16)))


	# XOR error string with cipherText
		error, errorLoc = createErrors(errorCount)
		XTS_Encrypt_Tweaked = byte_xor(XTS_Encrypt_Tweaked, error)
		Plain_Encrypt = byte_xor(Plain_Encrypt, error)



	#AES XTS Decrypt
		XTS_Decrypt_Tweaked = byte_xor(Tweak, XTS_Encrypt_Tweaked)
		XTS_Decrypt = AES_ECB.decrypt(XTS_Decrypt_Tweaked)
		XTS_PlainText = byte_xor(Tweak, XTS_Decrypt)

	#AES Decrypt
		Plain_Decrypt = AES_ECB.decrypt(Plain_Encrypt)	


	# AES Validity Check
		# print("PlainText_Original:\t{0:0128b}".format(int(PlainText_Original.hex(),16)))
		# print("Plain_Decrypt:\t\t{0:0128b}".format(int(Plain_Decrypt.hex(),16)))
		# print("XTS_PlainText:\t\t{0:0128b}".format(int(XTS_PlainText.hex(),16)))
		#assert PlainText_Original == Plain_Decrypt, "Plain AES Error"
		#assert PlainText_Original == XTS_PlainText, "AES-XTS Error"

	# Retriev new Error strings
		XTS_Error = byte_xor(PlainText_Original, XTS_PlainText)
		#print("XTS_Error:\t\t{0:0128b}".format(int(XTS_Error.hex(),16)))
		Plain_Error = byte_xor(PlainText_Original, Plain_Decrypt)
		#print("Plain_Error:\t\t{0:0128b}".format(int(Plain_Error.hex(),16)))

		XTS_Error_Loc = errorCounter(XTS_Error)
		Plain_Error_Loc = errorCounter(Plain_Error)

		print("{};{};{};{};".format(i, errorCount,len(XTS_Error_Loc), len(Plain_Error_Loc)))
		fout_SIMP.write("{};{};{};{};\n".format(i, errorCount,len(XTS_Error_Loc), len(Plain_Error_Loc)))
		fout.write("{};{};{};{};{};{};{};{};{};{};\n".format(i, errorCount, errorLoc, Key.hex(),Tweak.hex(),PlainText_Original.hex(), XTS_Error_Loc, len(XTS_Error_Loc),Plain_Error_Loc, len(Plain_Error_Loc)))

def AESTest_Controller(testNumber, iterations, errorCount):

	if not os.path.exists('data'):
		os.makedirs('data')

	print("data/AES-XTS-{}.csv".format(testNumber))
	print("data/AES-XTS-{}-SIMP.csv".format(testNumber))
	fout = open("data/AES-XTS-{}.csv".format(testNumber), "w")
	fout_SIMP = open("data/AES-XTS-{}-SIMP.csv".format(testNumber), "w")

	fout_SIMP.write("Round; Inserted Errors; XTS Errors; Plain Errors;\n")
	fout.write("Round; Inserted Errors; Inserted Error Loc; Key; Tweak; PlainText_Original; XTS Error Loc; XTS Errors;Plain Error Loc; Plain Errors;\n")

	for i in range(1,errorCount):
		AESTest(testNumber, iterations, i, fout, fout_SIMP)


def byte_xor(ba1, ba2):
    return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])

def createErrors(errorCount):
	errorLoc = random.sample(range(0,128),errorCount)
	errorLoc.sort(reverse=True)
	error = bytearray(16)
	error = int.from_bytes(error, byteorder='big')
	# print("error:\t\t{0:0128b}".format(error))
	# print("Plain_Encrypt:\t{0:0128b}".format(int(Plain_Encrypt.hex(),16)))
	
	k = 0
	for j in range(128,-1,-1):
		if k >= len(errorLoc):
			error = error << j+1
			break
		error = error << 1
		if j == errorLoc[k]:
			error += 1
			k += 1

	# print(errorLoc)
	# print("error:\t\t{0:0128b}".format(error))
	error = error.to_bytes(16, byteorder='big')
	# print("error:\t\t{0:0128b}".format(int(error.hex(),16)))
	return error, errorLoc

def errorCounter(errorString):
	errorString = int.from_bytes(errorString, byteorder='big')
	binary = bin(errorString)
	#print('\t\t\t{}'.format(binary[2:]))
	binary = (binary[2:])[::-1]
	#print('\t\t\t{}'.format(binary))

	error_Loc = [n for ones,n in zip(binary,range(128)) if ones=='1'] 

	#print(error_Loc,len(error_Loc))
	return error_Loc





if __name__ == "__main__":
    for i, arg in enumerate(sys.argv):
        print("Argument {}: {}".format(i,arg))

    if len(sys.argv) >= 3:
    	AESTest_Controller(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))






