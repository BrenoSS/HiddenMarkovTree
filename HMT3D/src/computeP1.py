import numpy as np
import dtcwt, math, copy, strc, sys
import nibabel as nib
from pdf import Rayleigh


class computeConditionalProb():
	def __init__(self, pyramid, PS, SI, ES):
		self.Pyramid = pyramid
		self.PS = PS
		self.SI = SI
		self.ES = ES
		self.nLevels = len(pyramid)
		self.nOrient = self.PS.shape[1]
		self.P1 = strc.createArrayProbOrient(self.Pyramid)
		self.P2 = strc.createArrayTransf(self.Pyramid)
		self.coeff_mag = self.getMag(pyramid)

	def getMag(self, pyramid):
		mag_pyr = []
		for scale in pyramid:
			mag_pyr.append(np.abs(scale))

		return mag_pyr

	def UP_Step(self, orient):
		BEC = strc.createArrayProb(self.Pyramid)
		BEP = strc.createArrayProb(self.Pyramid)
		BER = strc.createArrayProb(self.Pyramid)

		#Initialization
		scale = 0     # level of the leaf nodes
		WaveCoeff = self.coeff_mag[scale]

		pdf_S = Rayleigh(WaveCoeff[:,:,:,orient], self.SI[scale, orient, 0])
		pdf_L = Rayleigh(WaveCoeff[:,:,:,orient], self.SI[scale, orient, 1])

		norm = (pdf_S * self.PS[scale, orient, 0]) + (pdf_L * self.PS[scale, orient, 1])

		BEC[scale][:,:,:,0] = (pdf_S * self.PS[scale, orient, 0]) / norm
		BEC[scale][:,:,:,1] = (pdf_L * self.PS[scale, orient, 1]) / norm

		BEP[scale][:,:,:,0] = ((BEC[scale][:, :, :, 0] * self.ES[scale, orient, 0, 0]) / self.PS[scale, orient, 0]) + \
			((BEC[scale][:, :, :, 1] * self.ES[scale, orient, 0, 1]) / self.PS[scale, orient, 1]) 

		BEP[scale][:,:,:,1] = ((BEC[scale][:, :, :, 0] * self.ES[scale, orient, 1, 0]) / self.PS[scale, orient, 0]) + \
			((BEC[scale][:, :, :, 1] * self.ES[scale, orient, 1, 1]) / self.PS[scale, orient, 1])    

		#Induction
		for scale in range(1,self.nLevels):
			WaveCoeff = self.coeff_mag[scale]
			pdf_S = Rayleigh(WaveCoeff[:, :, :, orient], self.SI[scale, orient, 0])
			pdf_L = Rayleigh(WaveCoeff[:, :, :, orient], self.SI[scale, orient, 1])

			for i in range(WaveCoeff.shape[0]):
				ri = [i << 1, (i << 1)+1]
				for j in range(WaveCoeff.shape[1]):
					rj = [j << 1, (j << 1)+1]
					for k in range(WaveCoeff.shape[2]):
						rk = [k << 1, (k << 1)+1]

						CoeffBEP = BEP[scale-1]
						prodS = np.prod(CoeffBEP[ri[0]:ri[1]+1,rj[0]:rj[1]+1,rk[0]:rk[1]+1,0])
						prodL = np.prod(CoeffBEP[ri[0]:ri[1]+1,rj[0]:rj[1]+1,rk[0]:rk[1]+1,1])

						norm = (pdf_S[i, j, k] * prodS * self.PS[scale, orient, 0]) + (pdf_L[i, j, k] * prodL * self.PS[scale, orient, 1])

						BEC[scale][i, j, k, 0] = (pdf_S[i, j, k] * prodS * self.PS[scale, orient, 0]) / norm
						BEC[scale][i, j, k, 1] = (pdf_L[i, j, k] * prodL * self.PS[scale, orient, 1]) / norm

						BER[scale-1][ri[0]:ri[1]+1,rj[0]:rj[1]+1,rk[0]:rk[1]+1,:] = BEC[scale][i, j, k, :] /\
										BEP[scale-1][ri[0]:ri[1]+1,rj[0]:rj[1]+1,rk[0]:rk[1]+1,:]

			BEP[scale][:,:,:,0] = ((BEC[scale][:, :, :, 0] * self.ES[scale, orient, 0, 0]) / self.PS[scale, orient, 0]) + \
							((BEC[scale][:, :, :, 1] * self.ES[scale, orient, 0, 1]) / self.PS[scale, orient, 1]) 

			BEP[scale][:,:,:,1] = ((BEC[scale][:, :, :, 0] * self.ES[scale, orient, 1, 0]) / self.PS[scale, orient, 0]) + \
							((BEC[scale][:, :, :, 1] * self.ES[scale, orient, 1, 1]) / self.PS[scale, orient, 1]) 

		return (BEC, BEP, BER)

	def DOWN_Step(self, orient, BER):
		AL = strc.createArrayProb(self.Pyramid)
		scale = self.nLevels-1
		AL[scale][:, :, :, :] = 1

		for scale in range(self.nLevels-2, -1, -1):
			for i in range(AL[scale].shape[0]):
				iP = i >> 1
				for j in range(AL[scale].shape[1]):
					jP = j >> 1
					for k in range(AL[scale].shape[2]):
						kP = k >> 1
						AL[scale][i, j, k, 0] = ((AL[scale+1][iP, jP, kP, 0] * self.ES[scale, orient, 0, 0] * \
										BER[scale][i, j, k, 0]) + (AL[scale+1][iP, jP, kP, 1] * self.ES[scale, orient, 1, 0] *\
										BER[scale][i, j, k, 1])) / self.PS[scale, orient, 0]

						AL[scale][i, j, k, 1] = ((AL[scale+1][iP, jP, kP, 0] * self.ES[scale, orient, 0, 1] * \
										BER[scale][i, j, k, 0]) + (AL[scale+1][iP, jP, kP, 1] * self.ES[scale, orient, 1, 1] *\
										BER[scale][i, j, k, 1])) / self.PS[scale, orient, 1]
		return AL

	def computeProb(self, orient, BEC, BEP, BER, AL):

		for scale in range(self.nLevels):
			mult = AL[scale] * BEC[scale]
			self.P1[scale][:, :, :, orient, :] = mult

		for scale in range(self.nLevels-1):
			for i in range(self.P2[scale].shape[0]):
				iP = i >> 1
				for j in range(self.P2[scale].shape[1]):
					jP = j >> 1
					for k in range(self.P2[scale].shape[2]):
						kP = k >> 1
						for p in range(2):
							for f in range(2):
								self.P2[scale][i, j, k, orient, p, f] = (BEC[scale][i, j, k, f] * self.ES[scale, orient, p, f] \
									* AL[scale+1][iP, jP, kP, p] * BER[scale][i, j, k, p]) / self.PS[scale, orient, f]

	def computeProbFinal(self):
		for o in range(self.nOrient):
			BEC, BEP, BER = self.UP_Step(o)
			AL = self.DOWN_Step(o, BER)
			self.computeProb(o, BEC, BEP, BER, AL)
		return(self.P1,self.P2)