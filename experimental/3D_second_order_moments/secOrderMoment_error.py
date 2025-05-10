import numpy as np
import dtcwt_directionality


class wavelet2order():
	def __init__(self, pyramid, detected_points):
		self.pyramid = pyramid
		if(detected_points.shape[0] != 0):
			self.points_finest_scale = detected_points >> 1
		else:
			self.points_finest_scale = detected_points
		self.detected_points = detected_points
		self.elevation, self.azimuth = dtcwt_directionality.angle_infos()
		self.cos_azimuth = np.cos(self.azimuth)
		self.sin_azimuth = np.sin(self.azimuth)
		self.cos_elevation = np.cos(self.elevation)
		self.sin_elevation = np.sin(self.elevation)

	def sumScales(self, nLevels, coordI, coordJ, coordK, state, orient):
		somatorio = 0
		if (state == 0):
			states = self.x_orient
		elif(state == 1):
			states = self.y_orient
		elif(state == 2):
			states = self.z_orient
		for scale in range(nLevels):
			somatorio = somatorio + states[scale][coordI, coordJ, coordK, orient]
			coordI = coordI >> 1
			coordJ = coordJ >> 1
			coordK = coordK >> 1
		return somatorio


	def get_matrix2order(self):
		secondOrderMatrix = np.zeros(list(self.pyramid[0].shape[:3])+[3,3])
		
		x_orient = []
		y_orient = []
		z_orient = []
	
		for scale in self.pyramid:

			x_aux = scale * self.cos_elevation * self.cos_azimuth
			x_orient.append(x_aux)

			y_aux = scale * self.cos_elevation * self.sin_azimuth
			y_orient.append(y_aux)

			z_aux = scale * self.sin_azimuth
			z_orient.append(z_aux)


		self.x_orient = x_orient
		self.y_orient = y_orient
		self.z_orient = z_orient

		x = np.zeros(x_orient[0].shape)
		y = np.zeros(y_orient[0].shape)
		z = np.zeros(z_orient[0].shape)

		for orient in range(x_orient[0].shape[3]):
			for i, j, k in self.points_finest_scale:
				x[i, j, k, orient] = self.sumScales(len(x_orient), i, j, k, 0, orient)
				y[i, j, k, orient] = self.sumScales(len(y_orient), i, j, k, 1, orient)
				z[i, j, k, orient] = self.sumScales(len(z_orient), i, j, k, 2, orient)


		secondOrderMatrix[:, :, :, 0, 0] = np.sum(np.power(x, 2) * np.power(y, 0) * np.power(z, 0), axis=-1)
		secondOrderMatrix[:, :, :, 0, 1] = np.sum(np.power(x, 1) * np.power(y, 1) * np.power(z, 0), axis=-1)
		secondOrderMatrix[:, :, :, 0, 2] = np.sum(np.power(x, 1) * np.power(y, 0) * np.power(z, 1), axis=-1)
		secondOrderMatrix[:, :, :, 1, 0] = np.sum(np.power(x, 1) * np.power(y, 1) * np.power(z, 0), axis=-1)
		secondOrderMatrix[:, :, :, 1, 1] = np.sum(np.power(x, 0) * np.power(y, 2) * np.power(z, 0), axis=-1)
		secondOrderMatrix[:, :, :, 1, 2] = np.sum(np.power(x, 0) * np.power(y, 1) * np.power(z, 1), axis=-1)
		secondOrderMatrix[:, :, :, 2, 0] = np.sum(np.power(x, 1) * np.power(y, 0) * np.power(z, 1), axis=-1)
		secondOrderMatrix[:, :, :, 2, 1] = np.sum(np.power(x, 0) * np.power(y, 1) * np.power(z, 1), axis=-1)
		secondOrderMatrix[:, :, :, 2, 2] = np.sum(np.power(x, 0) * np.power(y, 0) * np.power(z, 2), axis=-1)

		return(secondOrderMatrix)


	def secondOrderMatrixAnalysis(self, error):

		output = np.zeros((self.pyramid[0].shape[0]*2,self.pyramid[0].shape[1]*2,self.pyramid[0].shape[2]*2))
		final_points = []

		if(self.points_finest_scale.shape[0] != 0):

			secondOrderMatrix = self.get_matrix2order()

			np.save("secondOrderMatrix", secondOrderMatrix)

			#secondOrderMatrix = np.load("secondOrderMatrix.npy")

			w, v = np.linalg.eigh(secondOrderMatrix)

			w = np.sort(w, axis=-1)

			minor_value = 0.5
			
			# error_mask = ((np.abs(w[:,:,:,0] - w[:,:,:,1]) < error) & (np.abs(w[:,:,:,0] - w[:,:,:,2]) < error) & (w[:,:,:,0] > minor_value))

			error_mask = ((np.abs(w[:,:,:,0] - w[:,:,:,1]) < error) & (np.abs(w[:,:,:,0] - w[:,:,:,2]) < error))

			out = np.zeros((secondOrderMatrix.shape[0], secondOrderMatrix.shape[1], secondOrderMatrix.shape[2]), dtype=bool)

			out[self.points_finest_scale[:,0],self.points_finest_scale[:,1],self.points_finest_scale[:,2]] = True

			out =  error_mask & out

			points_kept = np.argwhere(out == True)

			points_kept = np.asarray(points_kept)


			for point in self.detected_points:
				p_f_s = point >> 1
				if((points_kept == p_f_s).all(1).any()):
					final_points.append(point)

			final_points = np.asarray(final_points)

			if(final_points.shape[0] != 0):
				output[final_points[:,0],final_points[:,1],final_points[:,2]] = 1

		else:
			final_points = np.asarray(final_points)


		return(output, final_points)









