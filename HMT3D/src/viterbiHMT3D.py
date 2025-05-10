import numpy as np
import math, strc
from pdf import Rayleigh

class viterbiHMT3D():
	def __init__(self, pyramid):
		self.nOrient = 28
		self.nLevels = len(pyramid)
		self.pyramid = pyramid
		self.coeffMag = self.getMag(self.pyramid)

	def getMag(self, pyramid):
		mag_pyr = []
		for scale in pyramid:
			mag_pyr.append(np.abs(scale))

		return mag_pyr

	def readProbTreeModel(self, path):
		self.PS = np.load(path+"/PS.npy")
		self.ES = np.load(path+"/ES.npy")
		self.SI = np.load(path+"/SI.npy")

		self.fixParameters()

	def fixParameters(self):
		for scale in range(self.nLevels):
			for orient in range(self.nOrient):
				if(self.SI[scale][orient][0] > self.SI[scale][orient][1]):
					self.SI[scale][orient][0], self.SI[scale][orient][1] = self.SI[scale][orient][1], self.SI[scale][orient][0]
					self.PS[scale][orient][0], self.PS[scale][orient][1] = self.PS[scale][orient][1], self.PS[scale][orient][0]
					self.ES[scale][orient][1][1], self.ES[scale][orient][0][0] = self.ES[scale][orient][0][0], self.ES[scale][orient][1][1]
					self.ES[scale][orient][1][0], self.ES[scale][orient][0][1] = self.ES[scale][orient][0][1], self.ES[scale][orient][1][0]


	def viterbi(self):
		DEC = strc.createArrayProbOrient(self.pyramid)
		DEP = strc.createArrayProbOrient(self.pyramid)
		EPS = strc.createArrayProbOrient(self.pyramid)
		STATES = strc.createArrayShape(self.pyramid)

		#Initialization
		#delta computation for the leaf nodes
		scale = 0

		CoeffAbs = self.coeffMag[scale]

		DEC[scale][:,:,:,:,0] = Rayleigh(CoeffAbs, self.SI[scale][:,0])
		DEC[scale][:,:,:,:,1] = Rayleigh(CoeffAbs, self.SI[scale][:,1])

		#self.ES[scale][:,0,:] retorna uma matriz, para todas as orientações, com a linha 0
		mult_j0 = DEC[scale] * self.ES[scale][:,0,:]
		DEP[scale][:,:,:,:,0] = np.max(mult_j0,axis=-1)
		EPS[scale][:,:,:,:,0] = np.argmax(mult_j0,axis=-1)

		mult_j1 = DEC[scale] * self.ES[scale][:,1,:]
		DEP[scale][:,:,:,:,1] = np.max(mult_j1,axis=-1)
		EPS[scale][:,:,:,:,1] = np.argmax(mult_j1,axis=-1)

		#Induction
		for scale in range(1,self.nLevels):
			CoeffAbs = self.coeffMag[scale]

			pdf_S = Rayleigh(CoeffAbs, self.SI[scale][:,0])
			pdf_L = Rayleigh(CoeffAbs, self.SI[scale][:,1])


			for i in range(CoeffAbs.shape[0]):
				ri = [i << 1, (i << 1)+1]
				for j in range(CoeffAbs.shape[1]):
					rj = [j << 1, (j << 1)+1]
					for k in range(CoeffAbs.shape[2]):
						rk = [k << 1, (k << 1)+1]
						for orient in range(self.nOrient):

							prodS = np.prod(DEP[scale-1][ri[0]:ri[1]+1,rj[0]:rj[1]+1,rk[0]:rk[1]+1,orient,0])
							prodL = np.prod(DEP[scale-1][ri[0]:ri[1]+1,rj[0]:rj[1]+1,rk[0]:rk[1]+1,orient,1])
							
							DEC[scale][i,j,k,orient,0] = prodS * pdf_S[i,j,k,orient]
							DEC[scale][i,j,k,orient,1] = prodL * pdf_L[i,j,k,orient]


			mult_j0 = DEC[scale] * self.ES[scale][:,0,:]
			DEP[scale][:,:,:,:,0] = np.max(mult_j0,axis=-1)
			EPS[scale][:,:,:,:,0] = np.argmax(mult_j0,axis=-1)

			mult_j1 = DEC[scale] * self.ES[scale][:,1,:]
			DEP[scale][:,:,:,:,1] = np.max(mult_j1,axis=-1)
			EPS[scale][:,:,:,:,1] = np.argmax(mult_j1,axis=-1)

		#Termination
		scale = self.nLevels-1
		totProb = np.max(DEC[scale], axis=-1)

		STATES[scale] = np.argmax(DEC[scale], axis=-1)

		#DownWard tracking
		
		for scale in range(self.nLevels-2,-1,-1):
			for i in range(self.pyramid[scale].shape[0]):
				iP = i >> 1
				for j in range(self.pyramid[scale].shape[1]):
					jP = j >> 1
					for k in range(self.pyramid[scale].shape[2]):
						kP = k >> 1
						for orient in range(self.nOrient):
							stateParent = int(STATES[scale][iP,jP,kP,orient])
							STATES[scale][i,j,k,orient] = EPS[scale][i,j,k,orient,stateParent]

		return STATES


	def sumScales(self, coordI, coordJ, orient, states):
		somatorio = 0
		for scale in range(self.nLevels):
			somatorio = somatorio + states[scale][coordI][coordJ][orient]
			coordI = coordI >> 1
			coordJ = coordJ >> 1
		return somatorio

	def getStateSequence(self):
		return 0