import numpy as np
import dtcwt_directionality


class wavelet2order():
	def __init__(self, pyramid):
		self.pyramid = pyramid  #pyramid recebe os valores de magnitude
		self.elevation, self.azimuth = dtcwt_directionality.angle_infos()
		self.cos_azimuth = np.cos(self.azimuth)
		self.sin_azimuth = np.sin(self.azimuth)
		self.cos_elevation = np.cos(self.elevation)
		self.sin_elevation = np.sin(self.elevation)

	def sumScales(self, nLevels, coordI, coordJ, coordK, states):
		somatorio = 0
		for scale in range(nLevels):
			somatorio = somatorio + states[scale][coordI, coordJ,coordK]
			coordI = coordI >> 1
			coordJ = coordJ >> 1
			coordK = coordK >> 1
		return somatorio


	def get_matrix2order(self, threshold):
		secondOrderMatrix = np.zeros(list(self.pyramid[0].shape[:3])+[3,3])

		print(secondOrderMatrix.shape)

		#print(self.pyramid[0][0,0,0,0])

		#for (x, y), element in np.ndenumerate(self.pyramid[0][:,:,0]):
		
		x_scale = []
		y_scale = []
		z_scale = []

		for scale in self.pyramid:

			x_aux = np.sum((scale * self.cos_elevation) * (scale * self.cos_azimuth), axis=-1)
			#print(a_aux.shape)
			x_scale.append(x_aux)

			y_aux = np.sum((scale * self.cos_elevation) * (scale * self.sin_azimuth), axis=-1)
			#print(b_aux.shape)
			y_scale.append(y_aux)

			z_aux = np.sum(scale * self.sin_elevation, axis=-1)
			#print(c_aux.shape)
			z_scale.append(z_aux)

		x = np.zeros(x_scale[0].shape)
		y = np.zeros(y_scale[0].shape)
		z = np.zeros(z_scale[0].shape)

		for (i, j, k), element in np.ndenumerate(x_scale[0]):
			x[i, j, k] = self.sumScales(len(x_scale), i, j, k, x_scale)
			y[i, j, k] = self.sumScales(len(y_scale), i, j, k, y_scale)
			z[i, j, k] = self.sumScales(len(z_scale), i, j, k, z_scale)


			#secondOrderMatrix[i, j, k, 0, 0] = np.power(x[i, j, k])


		secondOrderMatrix[:, :, :, 0, 0] = np.power(x, 2) * np.power(y, 0) * np.power(z, 0)
		secondOrderMatrix[:, :, :, 0, 1] = np.power(x, 1) * np.power(y, 1) * np.power(z, 0)
		secondOrderMatrix[:, :, :, 0, 2] = np.power(x, 1) * np.power(y, 0) * np.power(z, 1)
		secondOrderMatrix[:, :, :, 1, 0] = np.power(x, 1) * np.power(y, 1) * np.power(z, 0)
		secondOrderMatrix[:, :, :, 1, 1] = np.power(x, 0) * np.power(y, 2) * np.power(z, 0)
		secondOrderMatrix[:, :, :, 1, 2] = np.power(x, 0) * np.power(y, 1) * np.power(z, 1)
		secondOrderMatrix[:, :, :, 2, 0] = np.power(x, 1) * np.power(y, 0) * np.power(z, 1)
		secondOrderMatrix[:, :, :, 2, 1] = np.power(x, 0) * np.power(y, 1) * np.power(z, 1)
		secondOrderMatrix[:, :, :, 2, 2] = np.power(x, 0) * np.power(y, 0) * np.power(z, 2)

		print(secondOrderMatrix[0,0,0])

		w, v = np.linalg.eigh(secondOrderMatrix)

		print(w.shape, v.shape)

		min_w = np.min(w,axis=-1)

		print("Max min: ", np.max(min_w))

		min_w_norm = (min_w - np.min(min_w)) / (np.max(min_w) - np.min(min_w))

		out = np.zeros(min_w_norm.shape)

		out[min_w_norm > threshold] = 1

		print(np.argwhere(out==1).shape[0])


		return(out)








