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


	def get_matrix2order(self, threshold):
		secondOrderMatrix = np.zeros(list(self.pyramid[0].shape[:3])+[3,3])

		print(secondOrderMatrix.shape)

		#print(self.pyramid[0][0,0,0,0])

		#for (x, y), element in np.ndenumerate(self.pyramid[0][:,:,0]):
		
		x_orient = []
		y_orient = []
		z_orient = []


		i_teste = 62
		j_teste = 62
		k_teste = 62

		i_teste_2 = 62
		j_teste_2 = 62
		k_teste_2 = 62		

		for scale in self.pyramid:

			print(scale[i_teste,j_teste,k_teste,0], self.cos_elevation[0], self.cos_azimuth[0], scale[i_teste,j_teste,k_teste,0] * self.cos_elevation[0] * self.cos_azimuth[0])
			# print(scale[i_teste,j_teste,k_teste,15], self.cos_elevation[15], self.cos_azimuth[15], scale[i_teste,j_teste,k_teste,15] * self.cos_elevation[15] * self.cos_azimuth[15])

			x_aux = scale * self.cos_elevation * self.cos_azimuth
			x_orient.append(x_aux)

			print(x_aux[i_teste,j_teste,k_teste,0])
			# print(x_aux[i_teste,j_teste,k_teste,15])

			y_aux = scale * self.cos_elevation * self.sin_azimuth
			y_orient.append(y_aux)

			z_aux = scale * self.sin_azimuth
			#print(c_aux.shape)
			z_orient.append(z_aux)

			i_teste = i_teste >> 1
			j_teste = j_teste >> 1
			k_teste = k_teste >> 1



		self.x_orient = x_orient
		self.y_orient = y_orient
		self.z_orient = z_orient

		x = np.zeros(x_orient[0].shape)
		y = np.zeros(y_orient[0].shape)
		z = np.zeros(z_orient[0].shape)

		print(x_orient[0].shape)
		for orient in range(x_orient[0].shape[3]):
			print("Orient: ",orient)
			for i in range(x_orient[0].shape[0]):
				for j in range(x_orient[0].shape[1]):
					for k in range(x_orient[0].shape[2]):
						x[i, j, k, orient] = self.sumScales(len(x_orient), i, j, k, 0, orient)
						y[i, j, k, orient] = self.sumScales(len(y_orient), i, j, k, 1, orient)
						z[i, j, k, orient] = self.sumScales(len(z_orient), i, j, k, 2, orient)



		# print(x_orient[0].shape)
		# for orient in range(x_orient[0].shape[3]):
		# 	print(orient)
		# 	print(x_orient[0][:,:,:,orient])
		# 	# for (i,j,k), element in np.ndenumerate(x_orient[0][:,:,:,orient]):
		# 	# 	print(i,j,k,element)

						

		print(x[i_teste_2,j_teste_2,k_teste_2,0])


		secondOrderMatrix[:, :, :, 0, 0] = np.sum(np.power(x, 2) * np.power(y, 0) * np.power(z, 0), axis=-1)
		secondOrderMatrix[:, :, :, 0, 1] = np.sum(np.power(x, 1) * np.power(y, 1) * np.power(z, 0), axis=-1)
		secondOrderMatrix[:, :, :, 0, 2] = np.sum(np.power(x, 1) * np.power(y, 0) * np.power(z, 1), axis=-1)
		secondOrderMatrix[:, :, :, 1, 0] = np.sum(np.power(x, 1) * np.power(y, 1) * np.power(z, 0), axis=-1)
		secondOrderMatrix[:, :, :, 1, 1] = np.sum(np.power(x, 0) * np.power(y, 2) * np.power(z, 0), axis=-1)
		secondOrderMatrix[:, :, :, 1, 2] = np.sum(np.power(x, 0) * np.power(y, 1) * np.power(z, 1), axis=-1)
		secondOrderMatrix[:, :, :, 2, 0] = np.sum(np.power(x, 1) * np.power(y, 0) * np.power(z, 1), axis=-1)
		secondOrderMatrix[:, :, :, 2, 1] = np.sum(np.power(x, 0) * np.power(y, 1) * np.power(z, 1), axis=-1)
		secondOrderMatrix[:, :, :, 2, 2] = np.sum(np.power(x, 0) * np.power(y, 0) * np.power(z, 2), axis=-1)

		print(secondOrderMatrix[i_teste_2,j_teste_2,k_teste_2])

		w, v = np.linalg.eigh(secondOrderMatrix)

		np.save("second_order_matrix", secondOrderMatrix)

		print(w.shape, v.shape)

		min_w = np.min(w,axis=-1)

		print("Max min: ", np.max(min_w))

		#min_w_norm = (min_w - np.min(min_w)) / (np.max(min_w) - np.min(min_w))

		out = np.zeros(min_w.shape)

		out[min_w > threshold] = 1

		print(np.argwhere(out==1).shape[0])

		return(out)








