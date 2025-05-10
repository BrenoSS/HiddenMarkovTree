import numpy as np
import dtcwt
import nibabel as nib
from 	scipy import ndimage as nd
import math
import copy
import strc
from pdf import Rayleigh
from HMT3Ddurand_v2 import HMT
import sys
import os
from computeP1 import computeConditionalProb

def computeWaveletTransform(data, nLevels):
	trans = dtcwt.Transform3d(biort='near_sym_b_bp', qshift='qshift_b_bp')
	Z = trans.forward(data, nlevels=nLevels)
	return Z.highpasses


def showUsage():
	print("Usage:\npython3 "+sys.argv[0]+" <path-to-image> <nr scales> <flag> (0 - treinamento / 1 - classificação / 2 - treinamento e classificação)")


def sumScales(nLevels, coordI, coordJ, coordK, states):
	somatorio = 0
	for scale in range(nLevels):
		somatorio = somatorio + states[scale][coordI, coordJ, coordK]
		coordI = coordI >> 1
		coordJ = coordJ >> 1
		coordK = coordK >> 1
	return somatorio


def checkPersistence(nLevels, states):
	summ = np.zeros(states[0].shape)
	print(summ.shape)

	for i in range(summ.shape[0]):
		for j in range(summ.shape[1]):
			for k in range(summ.shape[2]):
				summ[i, j, k] = sumScales(nLevels,i,j,k,states)

	return summ


nOrient = 28

program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

if(count != 3):
	showUsage()
	sys.exit()

path_image = sys.argv[1]
nLevels = int(sys.argv[2])
flag = int(sys.argv[3])


im = nib.load(path_image)
pyramid = computeWaveletTransform(im.get_data(), nLevels)


path_folder = "modelo/"
base=os.path.basename(path_image)
baseName = os.path.splitext(base)[0]
model_directory = path_folder + baseName + "/"


result_directory = "resultados/" + baseName + "/"

if((flag==0) or (flag==2)):
	if not os.path.exists(model_directory):
		os.makedirs(model_directory)

	tree = HMT(pyramid)
	tree.LearnModel()
	tree.saveModel(model_directory)

if((flag==1) or (flag==2)):
	if not os.path.exists(model_directory):
		print("There's no model files directory. It is needed to train the model before classifictation.")
		sys.exit()
	if not os.path.exists(result_directory):
		os.makedirs(result_directory)

	PS = np.load(model_directory+"/PS.npy")
	ES = np.load(model_directory+"/ES.npy")
	SI = np.load(model_directory+"/SI.npy")

	# with open("PS.txt",'w') as PS_txt:
	# 	PS_txt.write(str(PS))

	# with open("ES.txt",'w') as ES_txt:
	# 	ES_txt.write(str(ES))

	# with open("SI.txt",'w') as SI_txt:
	# 	SI_txt.write(str(SI))
		
	for scale in range(nLevels):
		for orient in range(nOrient):
			if(SI[scale][orient][0] > SI[scale][orient][1]):
				SI[scale][orient][0], SI[scale][orient][1] = SI[scale][orient][1], SI[scale][orient][0]
				PS[scale][orient][0], PS[scale][orient][1] = PS[scale][orient][1], PS[scale][orient][0]
				ES[scale][orient][1][1], ES[scale][orient][0][0] = ES[scale][orient][0][0], ES[scale][orient][1][1]
				ES[scale][orient][1][0], ES[scale][orient][0][1] = ES[scale][orient][0][1], ES[scale][orient][1][0]

	condProb = computeConditionalProb(pyramid, PS, SI, ES)
	P1, P2 = condProb.computeProbFinal()

	with open("P1.txt",'w') as P1_txt:
		P1_txt.write(str(P1))

	states = []

	P1_L_1 = P1[0][:,:,:,:,1] + 1
	prodP1_L = np.prod(P1_L_1,axis=-1)

	out_image = nib.Nifti1Image(prodP1_L, im.get_affine())
	nib.save(out_image, "output.nii")

	# for scale in range(nLevels):

	# 	WaveCoeff = pyramid[scale]
	# 	CoeffAbs = np.abs(WaveCoeff)

	# 	scale_state = np.zeros(CoeffAbs.shape[:3])

	# 	P1_L_1 = P1[scale][:,:,:,:,1] + 1
	# 	P1_S_1 = P1[scale][:,:,:,:,0] + 1

	# 	prodP1_L = np.prod(P1_L_1,axis=-1)
	# 	print(prodP1_L.shape)

	# 	prodP1_S = np.prod(P1_S_1,axis=-1)

	# 	scale_state[prodP1_L>prodP1_S] = 1

	# 	states.append(scale_state)


	# summ = checkPersistence(nLevels, states)
	# output = np.zeros(summ.shape)

	# output[summ==nLevels]=255
	# out_image = nib.Nifti1Image(output, im.get_affine())
	# nib.save(out_image, "output.nii")




