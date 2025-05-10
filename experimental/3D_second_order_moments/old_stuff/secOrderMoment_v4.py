import numpy as np
import dtcwt_directionality


class wavelet2order():
	def __init__(self, pyramid, detected_points):
		self.pyramid = pyramid  #pyramid recebe os valores de magnitude
		self.points_finest_scale = detected_points >> 1
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


	def get_matrix2order(self, threshold, tmax):
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
			for i, j, k in self.points_finest_scale:
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

		test_point = self.points_finest_scale[15]

		print(secondOrderMatrix[test_point[0],test_point[1],test_point[2]])

		w, v = np.linalg.eigh(secondOrderMatrix)

		w = np.sort(w, axis=-1)

		np.save("second_order_matrix", secondOrderMatrix)

		det = w[:,:,:,1] * w[:,:,:,2] + w[:,:,:,2] * w[:,:,:,0] +  w[:,:,:,0] * w[:,:,:,1]
		trace = np.sum(w,axis=-1)

		#dispersion = np.zeros((secondOrderMatrix.shape[0], secondOrderMatrix.shape[1], secondOrderMatrix.shape[2]))

		dispersion = np.divide((np.power(trace,3)), det, out=np.zeros_like(det), where=det!=0) #(np.power(trace,3)) / det

		print(dispersion[det>0])
		print(np.max(dispersion))

		limiar = np.power(2*tmax + 1,3)/np.power(tmax,2)

		print("Limiar value: ",limiar)

		print("Shape do det: ", det.shape)

		# min_w = np.min(w,axis=-1)

		# print("Max min: ", np.max(min_w))

		#min_w_norm = (min_w - np.min(min_w)) / (np.max(min_w) - np.min(min_w))

		out = np.ones((secondOrderMatrix.shape[0], secondOrderMatrix.shape[1], secondOrderMatrix.shape[2]))

		# out[min_w > threshold] = 1

		out[det <= 0] = 0
		out[trace * det <= 0] = 0

		out[dispersion > tmax] = 0
		#out[dispersion == 0] = 0

		points_kept = np.argwhere(out==1)

		print(points_kept.shape[0])

		final_points = []

		for point in self.detected_points:
			if (point >> 1) in points_kept:
				final_points.append(point)

		final_points = np.asarray(final_points)

		output = np.zeros((self.pyramid[0].shape[0]*2,self.pyramid[0].shape[1]*2,self.pyramid[0].shape[2]*2))
		print(output.shape)

		output[final_points[:,0],final_points[:,1],final_points[:,2]] = 1


		print(final_points.shape[0])

		return(output)








