import numpy as np
import dtcwt, math, copy, strc, sys, os
import nibabel as nib
from pdf import Rayleigh
from HMT3Ddurand_v2 import HMT
from viterbiHMT3D import viterbiHMT3D

def computeWaveletTransform(data, nLevels):
	trans = dtcwt.Transform3d(biort='near_sym_b_bp', qshift='qshift_b_bp')
	Z = trans.forward(data, nlevels=nLevels)
	return Z.highpasses


def showUsage():
	print("Usage:\npython3 "+sys.argv[0]+" <path-to-image> <nr scales> <flag> (0 - treinamento / 1 - classificação / 2 - treinamento e classificação)")

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
	viterbi = viterbiHMT3D(pyramid)
	viterbi.readProbTreeModel(model_directory)
	states = viterbi.viterbi()
