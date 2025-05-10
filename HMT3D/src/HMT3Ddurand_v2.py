import numpy as np
import math
import copy
import strc
from pdf import Rayleigh
import sys
import os


MAXIMUM_ITERATIONS = 200
SMALL_NUMBER = 1E-3

DEBUG = True

class HMT():
    def __init__(self, pyramid):
        self.nOrient = 28
        self.nLevels = len(pyramid)
        self.Pyramid = pyramid
        self.coeff_mag = self.getMag(pyramid)


    def getMag(self, pyramid):
        mag_pyr = []
        for scale in pyramid:
            mag_pyr.append(np.abs(scale))

        return mag_pyr

    def createStructure(self):
        # Sigma from Rayleigh function to be estimated
        self.SI = strc.createArray2D(self.nLevels, self.nOrient)
        # States transiction matrix for each scale and each orientation
        self.ES = strc.createArrayMatrix(self.nLevels, self.nOrient)
        # Strutcure to represent the marginal distribution of each hidden variable
        self.PS = strc.createArray2D(self.nLevels, self.nOrient)
        #conditional probabilities from the EM algorithm
        self.P1 = strc.createArrayProbOrient(self.Pyramid)
        self.P2 = strc.createArrayTransf(self.Pyramid)


    def distOfHiddenStates(self):

        #Initialization
        self.PS[self.nLevels-1, :, :] = 0.5 #(nLevels - 1) corresponde ao nível do root

        # Induction
        for scale in range(self.nLevels-2,-1,-1):
            for orient in range(self.nOrient):
                self.PS[scale, orient] = np.matmul(self.PS[scale+1, orient], self.ES[scale, orient])

    def initializeParameters(self):

        for scale in range(self.nLevels):
            for orient in range(self.nOrient):
                rand = np.random.random()
                maxSubBand = np.max(self.coeff_mag[scale][:,:,:,orient])
                self.SI[scale, orient, 0] = rand * 0.5
                self.SI[scale, orient, 1] = maxSubBand

        self.ES[:,:,:,:] = 0.5
        self.distOfHiddenStates()

    def storeParameters(self):
        self.SIO = copy.deepcopy(self.SI)
        self.ESO = copy.deepcopy(self.ES)
        self.PSO = copy.deepcopy(self.PS)

    def UP_Step(self, orient):
        #Beta Child Variable, indexed by scale, xyz coordinates and orientation
        BEC = strc.createArrayProb(self.Pyramid)
        #Beta Parent Variable, indexed by scale, xyz coordinates and orientation
        BEP = strc.createArrayProb(self.Pyramid)
        #Beta Regularized Variable, indexed by scale, xyz coordinates and orientation
        BER = strc.createArrayProb(self.Pyramid)

        #Initialization
        scale = 0     # level of the leaf nodes
        WaveCoeff = self.coeff_mag[scale]

        pdf_S = Rayleigh(WaveCoeff[:,:,:,orient], self.SI[scale, orient, 0])
        pdf_L = Rayleigh(WaveCoeff[:,:,:,orient], self.SI[scale, orient, 1])

        norm = (pdf_S * self.PS[scale, orient, 0]) + (pdf_L * self.PS[scale, orient, 1])

        BEC[scale][:,:,:,0] = (pdf_S * self.PS[scale, orient, 0]) / norm
        BEC[scale][:,:,:,1] = (pdf_L * self.PS[scale, orient, 1]) / norm

        # for i in range(WaveCoeff.shape[0]):
        #     for j in range(WaveCoeff.shape[1]):
        #         for k in range(WaveCoeff.shape[2]):
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
                                self.P2[scale][i, j, k, orient, f, p] = (BEC[scale][i, j, k, f] * self.ES[scale, orient, p, f] \
                                                            * AL[scale+1][iP, jP, kP, p] * BER[scale][i, j, k, p]) / self.PS[scale, orient, f]


    def Maximization(self,orient):

        #atualizacao dos valores de probabilidade:
        for scale in range(self.nLevels):
            sumP = np.sum(self.P1[scale][:, :, :, orient], axis=(0,1,2))
            #numElem = math.pow(int(self.OriginalSize[0]/math.pow(2, scale)),2)
            #numElem = self.P1[scale][:,:,orient].size / sumP.size
            numElem = self.P1[scale].shape[0] * self.P1[scale].shape[1] * self.P1[scale].shape[2]
            self.PS[scale,orient,:] = sumP / numElem

        for scale in range(self.nLevels-1):
            numElem = self.P2[scale].shape[0] * self.P2[scale].shape[1] * self.P2[scale].shape[2]
            sumESIn = np.sum(self.P2[scale][:, :, :, orient],axis=(0,1,2))
            for i in range(2):
                for j in range(2):
                    self.ES[scale, orient, i, j] = sumESIn[i, j] / (numElem * self.PS[scale+1, orient, j])

        for scale in range(self.nLevels):
            WaveCoeffR = np.real(self.Pyramid[scale][:, :, :, orient])
            WaveCoeffI = np.imag(self.Pyramid[scale][:, :, :, orient])

            numElem = self.Pyramid[scale].shape[0] * self.Pyramid[scale].shape[1] * self.Pyramid[scale].shape[2]

            tempS = numElem * self.PS[scale, orient, 0]
            tempL = numElem * self.PS[scale, orient, 1]

            tsis_re = np.sum(np.power(WaveCoeffR,2) * self.P1[scale][:,:,:,orient,0]) / tempS
            tsil_re = np.sum(np.power(WaveCoeffR,2) * self.P1[scale][:,:,:,orient,1]) / tempL

            tsis_im = np.sum(np.power(WaveCoeffI,2) * self.P1[scale][:,:,:,orient,0]) / tempS
            tsil_im = np.sum(np.power(WaveCoeffI,2) * self.P1[scale][:,:,:,orient,1]) / tempL


            self.SI[scale, orient, 0] = np.sqrt(tsis_re * tsis_im)
            self.SI[scale, orient, 1] = np.sqrt(tsil_re * tsil_im)

            # CoeffAbs = np.abs(self.Pyramid[scale][:,:,orient])
            # Coeff2 = np.power(CoeffAbs,2)

            # numElem = self.Pyramid[scale].shape[0] * self.Pyramid[scale].shape[1]
            # tempS = numElem * self.PS[scale][orient][0]
            # tempL = numElem * self.PS[scale][orient][1]

            # self.SI[scale,orient,0] = np.sum(Coeff2 * self.P1[scale][:,:,orient,0]) / tempS
            # self.SI[scale,orient,1] = np.sum(Coeff2 * self.P1[scale][:,:,orient,1]) / tempL


    def Convergency(self):

        t1 = np.abs((self.PS[:,:,0]-self.PSO[:,:,0])) / (self.PS[:,:,0]+self.PSO[:,:,0])
        t2 = np.abs((self.PS[:,:,1]-self.PSO[:,:,1])) / (self.PS[:,:,1]+self.PSO[:,:,1])

        pserr = max(np.max(t1),np.max(t2))

        t1 = np.abs((self.SI[:,:,0]-self.SIO[:,:,0])) / (self.SI[:,:,0]+self.SIO[:,:,0])
        t2 = np.abs((self.SI[:,:,1]-self.SIO[:,:,1])) / (self.SI[:,:,1]+self.SIO[:,:,1])

        sierr = max(np.max(t1),np.max(t2))

        t1 = np.abs((self.ES[:,:,0,0]-self.ESO[:,:,0,0])) / (self.ES[:,:,0,0]+self.ESO[:,:,0,0])
        t2 = np.abs((self.ES[:,:,0,1]-self.ESO[:,:,0,1])) / (self.ES[:,:,0,1]+self.ESO[:,:,0,1])
        t3 = np.abs((self.ES[:,:,1,0]-self.ESO[:,:,1,0])) / (self.ES[:,:,1,0]+self.ESO[:,:,1,0])
        t4 = np.abs((self.ES[:,:,1,1]-self.ESO[:,:,1,1])) / (self.ES[:,:,1,1]+self.ESO[:,:,1,1])

        eserr  = max(np.max(t1),np.max(t2))
        eserr  = max(eserr, np.max(t3))
        eserr  = max(eserr, np.max(t4))

        errval = max(pserr, sierr, eserr)

        return errval


    def saveModel(self, path):
    	np.save(path+'/P1', self.P1)
    	np.save(path+'/P2', self.P2)

    	np.save(path+'/PS', self.PS)
    	np.save(path+'/ES', self.ES)
    	np.save(path+'/SI', self.SI)

    def LearnModel(self):
        noIter = 0
        err2 = sys.maxsize
        err = 100

        self.createStructure()
        self.initializeParameters()

        it = 0
        small_error = sys.maxsize
        SMALL_NUMBER = 1E-2

        if not os.path.exists("modeloSNUpdate"):
            os.makedirs("modeloSNUpdate")

        while(err > SMALL_NUMBER):
            self.storeParameters()

            print("Iter: ", it)

            err1 = err2

            for o in range(self.nOrient):
                #Expectation
                BEC, BEP, BER = self.UP_Step(o)
                AL = self.DOWN_Step(o, BER)
                #Maximization
                self.computeProb(o, BEC, BEP, BER, AL)
                self.Maximization(o)


            err2 = self.Convergency()
            it +=1

            err = np.abs(err1 - err2)
            print(err)

            if(err < small_error):
                small_error = err
                np.save('modeloSNUpdate/PS', self.PS)
                np.save('modeloSNUpdate/ES', self.ES)
                np.save('modeloSNUpdate/SI', self.SI)
                print("Saving parameters from resulting convergency error ", small_error)

            if(it == 150):
                SMALL_NUMBER = small_error
                print("New SMALL_NUMBER: ", SMALL_NUMBER)
