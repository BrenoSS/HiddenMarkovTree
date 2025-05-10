import numpy as np
import math
from pdf import Rayleigh
import strc
import cv2

class detector():
	def __init__(self, nOrient, nLevels, pyramid):
		self.nOrient = nOrient
		self.nLevels = nLevels
		self.pyramid = pyramid

	def readProbTreeModel(self, path):
		self.PS = np.load(path+"PS.npy")
		self.ES = np.load(path+"ES.npy")
		self.SI = np.load(path+"SI.npy")
		with open("saidas/SI_original.txt","w") as siOriginal:
			siOriginal.write(str(self.SI))
		siOriginal.close()
		with open("saidas/PS_original.txt","w") as psOriginal:
			psOriginal.write(str(self.PS))
		psOriginal.close()
		with open("saidas/ES_original.txt","w") as esOriginal:
			esOriginal.write(str(self.ES))
		esOriginal.close()
		self.fixParameters()

	def fixParameters(self):
		for scale in range(self.nLevels):
			for orient in range(self.nOrient):
				if(self.SI[scale][orient][0] > self.SI[scale][orient][1]):
					self.SI[scale][orient][0], self.SI[scale][orient][1] = self.SI[scale][orient][1], self.SI[scale][orient][0]
					self.PS[scale][orient][0], self.PS[scale][orient][1] = self.PS[scale][orient][1], self.PS[scale][orient][0]
					self.ES[scale][orient][1][1], self.ES[scale][orient][0][0] = self.ES[scale][orient][0][0], self.ES[scale][orient][1][1]
					self.ES[scale][orient][1][0], self.ES[scale][orient][0][1] = self.ES[scale][orient][0][1], self.ES[scale][orient][1][0]
		with open("saidas/SI_trocado.txt","w") as siTrocado:
			siTrocado.write(str(self.SI))
		siTrocado.close()

		with open("saidas/PS_trocado.txt","w") as psTrocado:
			psTrocado.write(str(self.PS))
		psTrocado.close()

		with open("saidas/ES_trocado.txt","w") as esTrocado:
			esTrocado.write(str(self.ES))
		esTrocado.close()


	def viterbi(self):
		DEC = strc.createArrayProbOrient(self.pyramid)
		DEP = strc.createArrayProbOrient(self.pyramid)
		EPS = strc.createArrayProbOrient(self.pyramid)
		STATES = strc.createArrayShape(self.pyramid)

		for eps in EPS:
			print("Shape EPS ", eps.shape)

		for state in STATES:
			print(state.shape)

		#Initialization
		#delta computation for the leaf nodes
		scale = 0
		WaveCoeff = self.pyramid[scale]

		CoeffAbs = np.abs(WaveCoeff)

		DEC[scale][:,:,:,0] = Rayleigh(CoeffAbs, self.SI[scale][:,0])
		DEC[scale][:,:,:,1] = Rayleigh(CoeffAbs, self.SI[scale][:,1])

		#self.ES[scale][:,0,:] retorna uma matriz, para todas as orientações, com a linha 0
		mult_j0 = DEC[scale] * self.ES[scale][:,0,:]
		DEP[scale][:,:,:,0] = np.max(mult_j0,axis=-1)
		EPS[scale][:,:,:,0] = np.argmax(mult_j0,axis=-1)

		mult_j1 = DEC[scale] * self.ES[scale][:,1,:]
		DEP[scale][:,:,:,1] = np.max(mult_j1,axis=-1)
		EPS[scale][:,:,:,1] = np.argmax(mult_j1,axis=-1)

		# np.set_printoptions(threshold=np.nan)

		# print(EPS)

		# f = open("saidas/DEC.txt","w")
		# f.write(str(DEC))
		# f.close()

		# f = open("saidas/EPS.txt","w")
		# f.write(str(EPS))
		# f.close()

		# f = open("saidas/mult_j0.txt","w")
		# f.write(str(mult_j0))
		# f.close()

		# f = open("saidas/DEP.txt","w")
		# f.write(str(DEP))
		# f.close()

		# print(self.ES[scale])

		# DEP[scale]

		#Induction
		for scale in range(1,self.nLevels):
			WaveCoeff = self.pyramid[scale]

			CoeffAbs = np.abs(WaveCoeff)
			pdf_S = Rayleigh(CoeffAbs, self.SI[scale][:,0])
			pdf_L = Rayleigh(CoeffAbs, self.SI[scale][:,1])


			for i in range(CoeffAbs.shape[0]):
				ri = [i << 1, (i << 1)+1]
				for j in range(CoeffAbs.shape[1]):
					rj = [j << 1, (j << 1)+1]
					for orient in range(self.nOrient):

						prodS = np.prod(DEP[scale-1][ri[0]:ri[1]+1,rj[0]:rj[1]+1,orient,0])
						prodL = np.prod(DEP[scale-1][ri[0]:ri[1]+1,rj[0]:rj[1]+1,orient,1])

						DEC[scale][i,j,orient,0] = prodS * pdf_S[i][j][orient]
						DEC[scale][i,j,orient,1] = prodL * pdf_L[i][j][orient]


			mult_j0 = DEC[scale] * self.ES[scale][:,0,:]
			DEP[scale][:,:,:,0] = np.max(mult_j0,axis=-1)
			EPS[scale][:,:,:,0] = np.argmax(mult_j0,axis=-1)

			mult_j1 = DEC[scale] * self.ES[scale][:,1,:]
			DEP[scale][:,:,:,1] = np.max(mult_j1,axis=-1)
			EPS[scale][:,:,:,1] = np.argmax(mult_j1,axis=-1)

		#Termination
		scale = self.nLevels-1
		#print("DEC na escala mais grossa ", DEC[self.nLevels-1])
		totProb = np.max(DEC[scale], axis=-1)
		# print(DEC[self.nLevels-1].shape)
		# print(totProb.shape)
		#print(totProb)

		STATES[scale] = np.argmax(DEC[scale], axis=-1)

		# np.set_printoptions(threshold=np.nan)
		# print(STATES[self.nLevels-1])
		#DownWard tracking
		
		for scale in range(self.nLevels-2,-1,-1):
			for i in range(self.pyramid[scale].shape[0]):
				iP = i >> 1
				for j in range(self.pyramid[scale].shape[1]):
					jP = j >> 1
					for orient in range(self.nOrient):
						stateParent = int(STATES[scale][iP,jP,orient])
						STATES[scale][i,j,orient] = EPS[scale][i,j,orient,stateParent]

		return STATES


	def sumScales(self, coordI, coordJ, orient, states):
		somatorio = 0
		for scale in range(self.nLevels):
			somatorio = somatorio + states[scale][coordI][coordJ][orient]
			coordI = coordI >> 1
			coordJ = coordJ >> 1
		return somatorio

	def detect(self, states, path=False):
		summ = np.zeros(list(self.pyramid[0].shape))
		print(summ.shape)
		aux = np.zeros(list(self.pyramid[0].shape))

		for i in range(summ.shape[0]):
			for j in range(summ.shape[1]):
				for orient in range(self.nOrient):
					summ[i][j][orient] = self.sumScales(i,j,orient,states)

		aux = np.sum(summ,axis=-1)
		aux = aux / (self.nLevels * self.nOrient)

		output_img = np.zeros((summ.shape[0], summ.shape[1]))

		# for i in range(output_img.shape[0]):
		# 	for j in range(output_img.shape[1]):
		# 		if(aux[i][j] == 1):
		# 			output_img[i][j] = 255

		if(path):
			for scale in range(self.nLevels):
				for orient in range(self.nOrient):
					st = states[scale][:,:,orient]
					im = np.zeros((st.shape[0],st.shape[1]))
					im[st==1] = 255
					cv2.imwrite(path+"/"+str(scale)+"_"+str(orient)+".png",im)

		return summ