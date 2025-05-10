import numpy as np


ES = np.load("modelo/Template_T1_RAI/ES.npy")
PS = np.load("modelo/Template_T1_RAI/PS.npy")
SI = np.load("modelo/Template_T1_RAI/SI.npy")
P1 = np.load("modelo/Template_T1_RAI/P1.npy")
P2 = np.load("modelo/Template_T1_RAI/P2.npy")


nOrient = 1

with open("P1.txt",'w') as P1_txt:
	for scale in P1:
		print(scale.shape)
		P1_txt.write("\n============================\n")
		for orient in range(nOrient):
			# array_scale = scale[:,:,orient,:]
			# for (x,y), value in np.ndenumerate(array_scale):
			# 	P1_txt.write(str(value)+"\n")
			for x in range(scale.shape[0]):
				for y in range(scale.shape[1]):
					P1_txt.write(str(scale[x,y,orient])+"\n")

with open("P2.txt",'w') as P2_txt:
	for scale in P2:
		print(scale.shape)
		P2_txt.write("\n============================\n")
		for orient in range(nOrient):
			# array_scale = scale[:,:,orient,:]
			# for (x,y), value in np.ndenumerate(array_scale):
			# 	P1_txt.write(str(value)+"\n")
			for x in range(scale.shape[0]):
				for y in range(scale.shape[1]):
					P2_txt.write(str(scale[x,y,orient])+"\n")


with open("ES.txt",'w') as ES_txt:
	# for scale in ES:
	# 	print(scale.shape)
	# 	ES_txt.write(str(scale[0,0,0])+" "+str(scale[0,0,1])+"\n"+str(scale[0,1,0])+" "+str(scale[0,1,1])+"\n")
	ES_txt.write(str(ES))

with open("PS.txt",'w') as PS_txt:
	# for scale in PS:
	# 	print(scale.shape)
	# 	PS_txt.write(str(scale[0,0])+" "+str(scale[0,1])+"\n")
	PS_txt.write(str(PS))

with open("SI.txt",'w') as SI_txt:
	SI_txt.write(str(SI))