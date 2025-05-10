import 	numpy   as np
import 	nibabel as nib
from 	skimage.feature import peak_local_max
from skimage.transform import pyramid_expand
import copy

def keypoints3D(Pyramid, mask, nScales, alpha, beta, energy_threshold):

	nOrientations = Pyramid[-1].shape[3]

	# Accumulator
	summ = np.zeros(Pyramid[-1].shape[:3])

	# Main part of the method
	for scale in range(nScales-1, -1, -1):
		WaveCoeff = Pyramid[scale]
		prod = np.ones(WaveCoeff.shape[:3])

		for orient in range(nOrientations):
			# Compute the absolute of the complex coefficients
			coeff_abs = WaveCoeff[:,:,:,orient]

			# Then, weight the coefficients according to the scale
			prod = prod * np.power(coeff_abs, beta)

		summ = summ + np.power(alpha, scale) * prod 
		summ = pyramid_expand(summ, upscale=2, multichannel=False)


	# Perform detection of salient points only within the mask region
	summ[ mask == 0 ] = 0.0

	#save the norm energy map from the brain region
	energyMap = copy.deepcopy(summ)

	# Normalize energy map to the 0-1 range and then apply energy thredhold
	summ = (summ - np.min(summ))/(np.max(summ) - np.min(summ))

	#print(energy_threshold)
	summ[ summ < energy_threshold ] = 0.0 

	# Detect local maxima in the energy map
	neighbor_26  = np.ones((3,3,3))
	#neighbor_124  = np.ones((5,5,5))
	#neighbor_342 = np.ones((7,7,7))
	local_maxima = peak_local_max(summ, footprint=neighbor_26, indices=False, exclude_border=1)

	# Salving index of the detected points on the finest scale
	#np.set_printoptions(threshold=np.nan)
	points = np.argwhere(local_maxima == True)
	#np.save('points', points)

	# points_parent = points >> 1
	# np.save("points_parent", points_parent)

	# Overlay detected salient points on the original image and save
	summ = np.ones(mask.shape[:3])
	summ = summ * local_maxima

	return (summ, points, energyMap)
