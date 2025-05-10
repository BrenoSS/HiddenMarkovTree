import numpy as np
import dtcwt
import cv2
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
	print("Usage:\npython3 "+sys.argv[0]+" <path-to-image> <nr scales> <tr_type>")

nOrient = 6


program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

if(count != 3):
	showUsage()
	sys.exit()


path_image = sys.argv[1]
nLevels = int(sys.argv[2])
tr_type = int(sys.argv[3])


im = cv2.imread(path_image)
gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#gray_image_resized = cv2.resize(gray_image, (0,0), fx=2, fy=2) 
pyramid = computeWaveletTransform(gray_image,nLevels,tr_type)

#output_image = np.zeros((int(im.shape[0]/2),int(im.shape[1]/2)))

if(tr_type == 1):
	subFolder = "ajuste45_135/"
else:
	subFolder = "sem_ajuste/"

path_folder = "DTCWT-subBandImages/"
base=os.path.basename(path_image)
directory = path_folder + os.path.splitext(base)[0] + "/" + subFolder
if not os.path.exists(directory):
	os.makedirs(directory)


for scale in range(nLevels):
	coeff = np.abs(pyramid[scale])
	coeff_norm = (coeff - np.min(coeff)) / (np.max(coeff)- np.min(coeff))
	for orient in range(nOrient):
		output_image = coeff_norm[:,:,orient]

		output_image = output_image * 255

		cv2.imwrite(directory+"/"+str(scale)+"_"+str(orient)+".png",output_image)


