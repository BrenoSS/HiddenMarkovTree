import 	nibabel as nib
import 	dtcwt
import numpy as np
import keypoints3D as kp
import sys
import os
from secOrderMoment_eigenRelation import wavelet2order

def computeWaveletTransform(data, nLevels):
    trans = dtcwt.Transform3d(biort='near_sym_b_bp', qshift='qshift_b_bp')
    Z = trans.forward(data, nlevels=nLevels)
    return Z.highpasses


def showUsage():
	print("Usage:\npython3 "+sys.argv[0]+" <path-to-image> <nro scales> <alpha> <beta> <energy_threshold>")


if __name__ == '__main__':

	program_name = sys.argv[0]
	arguments = sys.argv[1:]
	count = len(arguments)

	if(count != 5):
		showUsage()
		sys.exit()

	path_image = sys.argv[1]
	nScales = int(sys.argv[2])
	alpha = float(sys.argv[3])
	beta = float(sys.argv[4])
	energy_threshold = float(sys.argv[5])


	path, file = os.path.split(path_image)
	base=os.path.basename(file)
	baseName = os.path.splitext(base)[0]
	output_filename = baseName + '_Landmarks.nii'
	energy_filename = baseName + '_Energy.nii'
	points_filename = baseName

	mask_filename   = path + "/" + baseName + '_mask.nii'
	img = nib.load(path_image)
	img_mask = nib.load(mask_filename)
	mask = img_mask.get_data()

	print("Transformada Wavelet da imagem de entrada")

	pyramid = computeWaveletTransform(img.get_data(), nScales)

	coeff_mag = []

	for scale in pyramid:
		scale_abs = np.abs(scale)
		scale_norm = (scale_abs - np.min(scale_abs)) / (np.max(scale_abs) - np.min(scale_abs))
		coeff_mag.append(scale_norm)

	# np.save("coeff_mag", coeff_mag)

	# coeff_mag = np.load("coeff_mag.npy")


	#print(coeff_mag[0][62,62,62,0], coeff_mag2[0][62,62,62,0])

	# alpha   = 0.25                                              
	# beta    = pow(alpha, nScales)        
	# energy_threshold = 1.0-alpha

	print("Detecção dos pontos")

	print("Nscales: ", nScales, "\nAlpha: ", alpha, "\nBeta: ", beta, "\nEnergy_threshold: ", energy_threshold)

	summ, detected_points, energyMap = kp.keypoints3D(pyramid, mask, nScales, alpha, beta, energy_threshold)

	# np.save("summ", summ)

	# np.save("detected_points", detected_points)

	# summ = np.load("summ.npy")
	# detected_points = np.load("detected_points.npy")


	print("Número de pontos detectados - (Kingsbury): ", detected_points.shape[0])

	moment3d = wavelet2order(coeff_mag, detected_points)

	out, points_kept = moment3d.secondOrderMatrixAnalysis(0.25)

	print("Número de pontos após a análise da matriz de momentos: ", points_kept.shape[0])


	result_directory = "detct_result/" + baseName + "/"
	if not os.path.exists(result_directory):
		os.makedirs(result_directory)

	# np.save(result_directory + points_filename, detected_points)

	out_image = nib.Nifti1Image(summ, img.get_affine())
	nib.save(out_image, result_directory + output_filename)

	# energy_image = nib.Nifti1Image(energyMap, img.get_affine())
	# nib.save(energy_image, result_directory + energy_filename)

	out_image2 = nib.Nifti1Image(out, img.get_affine())
	nib.save(out_image2, result_directory + "eigenvalue_analysis.nii")