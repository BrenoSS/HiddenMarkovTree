import numpy as np
import sys

def showUsage():
	print("Usage:\npython3 "+sys.argv[0]+" <path-to-model> <filename-txt>")

nOrient = 6

program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

if(count != 2):
	showUsage()
	sys.exit()

path = sys.argv[1]
filename = sys.argv[2]

SI = np.load(path+"/SI.npy")
ES = np.load(path+"/ES.npy")
PS = np.load(path+"/PS.npy")

nLevels = SI.shape[0]

with open(filename,'w') as parameters:
	for scale in range(nLevels-1,-1,-1):
		for orient in range(nOrient):
			parameters.write(str(ES[scale,orient,0,0]) + " " + \
							 str(ES[scale,orient,0,1]) + " " + \
							 str(ES[scale,orient,1,1]) + " " + \
							 str(ES[scale,orient,1,0]) + " " + \
							 str(PS[scale,orient,0]) + " " + \
							 str(PS[scale,orient,1]) + " " + \
							 str(SI[scale,orient,0]) + " " + \
							 str(SI[scale,orient,1]) + "\n")
