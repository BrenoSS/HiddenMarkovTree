import numpy as np
import dtcwt
import nibabel as nib
import math
import copy
import strc
from pdf import Rayleigh
import cv2
from HMT2Ddurand_v2 import HMT
from viterbiHMT import detector
import sys
import os

def computeWaveletTransform(data,nLevels,tr_type):
	if(tr_type==1):
		trans = dtcwt.Transform2d(biort='near_sym_b_bp', qshift='qshift_b_bp')
	else:
		trans = dtcwt.Transform2d()
	Z = trans.forward(data, nlevels=nLevels)
	return Z.highpasses


def showUsage():
	print("Usage:\npython3 "+sys.argv[0]+" <path-to-image> <nr scales> <tr_type> (0 - sem ajuste / 1 - ajuste 45,135) <flag> (0 - treinamento / 1 - classificação / 2 - treinamento e classificação)")

nOrient = 6

program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

if(count != 4):
	showUsage()
	sys.exit()

path_image = sys.argv[1]
nLevels = int(sys.argv[2])
tr_type = int(sys.argv[3])
flag = int(sys.argv[4])


im = cv2.imread(path_image)
gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
pyramid = computeWaveletTransform(gray_image, nLevels, tr_type)


if(tr_type == 1):
	subFolder = "ajuste45_135/"
else:
	subFolder = "sem_ajuste/"


path_folder = "modelo/"
base=os.path.basename(path_image)
baseName = os.path.splitext(base)[0]
model_directory = path_folder + baseName + "/" + subFolder


result_directory = "resultados/" + baseName + "/" + subFolder


if((flag==0) or (flag==2)):
	if not os.path.exists(model_directory):
		os.makedirs(model_directory)

	tree = HMT(nOrient, nLevels, pyramid)
	tree.LearnModel()
	tree.saveModel(model_directory)

if((flag==1) or (flag==2)):
	if not os.path.exists(model_directory):
		print("There's no model files directory. It is needed to train the model before classifictation.")
		sys.exit()
	if not os.path.exists(result_directory):
		os.makedirs(result_directory)
	detector = detector(nOrient, nLevels, pyramid)
	detector.readProbTreeModel(model_directory)
	states = detector.viterbi()
	summ = detector.detect(states,result_directory)

	for orient in range(summ.shape[-1]):
		output_img = np.zeros((summ.shape[0],summ.shape[1]))
		output_img[summ[:,:,orient]==nLevels] = 255
		cv2.imwrite(result_directory+"/out"+str(orient)+".jpg",output_img)

		output = np.zeros((summ.shape[0],summ.shape[1]))

		for orient in range(summ.shape[-1]):
			output[summ[:,:,orient]==nLevels] +=1

		output[output==nOrient] = 255
		cv2.imwrite(result_directory+"/final_output.jpg",output)
