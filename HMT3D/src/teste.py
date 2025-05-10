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
	print("Usage:\npython3 "+sys.argv[0]+" <path-to-image> <nr scales>")


nOrient = 28

program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

if(count != 2):
	showUsage()
	sys.exit()

path_image = sys.argv[1]
nLevels = int(sys.argv[2])


im = nib.load(path_image)
pyramid = computeWaveletTransform(im.get_data(), nLevels)

print(pyramid[0].shape)