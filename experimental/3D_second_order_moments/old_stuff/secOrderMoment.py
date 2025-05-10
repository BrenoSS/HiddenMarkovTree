import numpy as np
import dtcwt_directionality


class wavelet2order():
	def __init__(self, pyramid, detected_points):
		self.pyramid = pyramid  #pyramid recebe os valores de magnitude
		self.detected_points = detected_points
		self.elevation, self.azimuth = dtcwt_directionality.angle_infos()
		self.cos_azimuth = np.cos(self.azimuth)
		self.sin_azimuth = np.sin(self.azimuth)
		self.cos_elevation = np.cos(self.elevation)
		self.sin_elevation = np.sin(self.elevation)


	def get_matrix2order(self, point):
		matrix2order = np.zeros((3,3))

		n_orient = len(self.azimuth)

		x = []
		y = []
		z = []

		for orient in range(n_orient):
			x_orient = 0
			y_orient = 0
			z_orient = 0
			for scale in self.pyramid:
				# x_orient+= scale[point[0], point[1], point[2], orient] * self.cos_elevation[orient] * self.cos_azimuth[orient]
				# y_orient+= scale[point[0], point[1], point[2], orient] * self.cos_elevation[orient] * self.sin_azimuth[orient]
				# z_orient+= scale[point[0], point[1], point[2], orient] * self.sin_elevation[orient]
				x_orient+= scale[point[0], point[1], point[2], orient] * self.cos_azimuth[orient] * self.cos_elevation[orient]
				y_orient+= scale[point[0], point[1], point[2], orient] * self.cos_azimuth[orient] * self.sin_elevation[orient]
				z_orient+= scale[point[0], point[1], point[2], orient] * self.sin_azimuth[orient]
				point = point >> 1
			x.append(x_orient)
			y.append(y_orient)
			z.append(z_orient)

		x = np.asarray(x)
		y = np.asarray(y)
		z = np.asarray(z)

		matrix2order[0,0] = np.sum(np.power(x,2) * np.power(y,0) * np.power(z,0))
		matrix2order[0,1] = np.sum(np.power(x,1) * np.power(y,1) * np.power(z,0))
		matrix2order[0,2] = np.sum(np.power(x,1) * np.power(y,0) * np.power(z,1))
		matrix2order[1,0] = np.sum(np.power(x,1) * np.power(y,1) * np.power(z,0))
		matrix2order[1,1] = np.sum(np.power(x,0) * np.power(y,2) * np.power(z,0))
		matrix2order[1,2] = np.sum(np.power(x,0) * np.power(y,1) * np.power(z,1))
		matrix2order[2,0] = np.sum(np.power(x,1) * np.power(y,0) * np.power(z,1))
		matrix2order[2,1] = np.sum(np.power(x,0) * np.power(y,1) * np.power(z,1))
		matrix2order[2,2] = np.sum(np.power(x,0) * np.power(y,0) * np.power(z,2))

		return(matrix2order)


	def computeSecondOrderMatrix(self, threshold):

		salient_points = []

		for point in self.detected_points:
			#print(point)
			finest_scale_point = point >> 1
			#print(finest_scale_point)
			#print(self.pyramid[0][finest_scale_point[0], finest_scale_point[1], finest_scale_point[2],0])

			matrix2order = self.get_matrix2order(finest_scale_point)
			#print(matrix2order)

			w, v = np.linalg.eig(matrix2order)

			#print(w)

			if(np.min(w) > threshold):
				salient_points.append(point)


		return salient_points

		#return matrix2order








