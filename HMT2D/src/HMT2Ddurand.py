import numpy as np
import dtcwt
import nibabel as nib
import math
import copy
import strc
from pdf import Rayleigh
import cv2
import sys

def computeWaveletTransform(data,nLevels):
    trans = dtcwt.Transform2d()
    Z = trans.forward(data, nlevels=nLevels)
    return Z.highpasses


MAXIMUM_ITERATIONS = 200
SMALL_NUMBER = 1E-3

DEBUG = True

class HMT():
    def __init__(self, nOrient, nLevels, pyramid):
        self.nOrient = nOrient
        self.nLevels = nLevels
        self.Pyramid = pyramid
        #OS = list(self.Pyramid[0].shape)
        #self.OriginalSize = OS[:-1]

    def getCoordParent(self,coordChild):
        return math.floor( coordChild/2 )

    def getRangeChildren(self, CoordParent):
        return [2*CoordParent,2*CoordParent+1]

    def createStructure(self):
        # Sigma from Rayleigh function to be estimated
        self.SI = strc.createArray2D(self.nLevels, self.nOrient)
        # States transiction matrix for each scale and each orientation
        self.ES = strc.createArrayMatrix(self.nLevels, self.nOrient)
        # Strutcure to represent the marginal distribution of each hidden variable
        # OriginalSize = list(self.Pyramid[0].shape)
        # OriginalSize = OriginalSize[:-1]
        self.PS = strc.createArray2D(self.nLevels, self.nOrient)
        #conditional probabilities from the EM algorithm
        self.P1 = strc.createArrayProbOrient(self.Pyramid)
        self.P2 = strc.createArrayTransf(self.Pyramid)


    def distOfHiddenStates(self):

        #Initialization
        self.PS[self.nLevels-1][:,:] = 0.5

        # Induction
        for scale in range(self.nLevels-2,-1,-1):
            for orient in range(self.nOrient):
                self.PS[scale][orient] = np.matmul(self.PS[scale+1][orient], self.ES[scale][orient])

    def initializeParameters(self):

        for scale in range(self.nLevels):
            for orient in range(self.nOrient):
                rand = np.random.random()
                maxSubBand = np.max(np.abs(self.Pyramid[scale][:,:,orient]))
                self.SI[scale][orient][0] = rand * 0.5
                self.SI[scale][orient][1] = maxSubBand
                #print(self.SI[i][j][0], " ", self.SI[i][j][1])

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
        scale = 0
        WaveCoeff = self.Pyramid[scale]

        pdf_S = Rayleigh(np.abs(WaveCoeff[:,:,orient]), self.SI[scale][orient][0])
        pdf_L = Rayleigh(np.abs(WaveCoeff[:,:,orient]), self.SI[scale][orient][1])

        norm = (pdf_S * self.PS[scale][orient][0]) + (pdf_L * self.PS[scale][orient][1])

        BEC[scale][:,:,0] = (pdf_S * self.PS[scale][orient][0]) / norm
        BEC[scale][:,:,1] = (pdf_L * self.PS[scale][orient][1]) / norm

        for i in range(WaveCoeff.shape[0]):
            for j in range(WaveCoeff.shape[1]):
                BEP[scale][i,j,:] = ((BEC[scale][i,j,0]*self.ES[scale][orient,0,:])/self.PS[scale][orient][0]) + \
                                ((BEC[scale][i,j,1]*self.ES[scale][orient,1,:])/self.PS[scale][orient][1])     

        #Induction
        for scale in range(1,self.nLevels):
            WaveCoeff = self.Pyramid[scale]
            pdf_S = Rayleigh(np.abs(WaveCoeff[:,:,orient]), self.SI[scale][orient][0])
            pdf_L = Rayleigh(np.abs(WaveCoeff[:,:,orient]), self.SI[scale][orient][1])
            # print(WaveCoeff.shape[0])
            # print(WaveCoeff.shape[1])
            # print(WaveCoeff[:,:,orient].size)
            # sys.exit()
            for i in range(WaveCoeff.shape[0]):
            	ri = [i << 1, (i << 1)+1]
            	for j in range(WaveCoeff.shape[1]):
            		rj = [j << 1, (j << 1)+1]

            		CoeffScale = BEP[scale-1]
            		prodS = np.prod(CoeffScale[ri[0]:ri[1]+1,rj[0]:rj[1]+1,0])
            		prodL = np.prod(CoeffScale[ri[0]:ri[1]+1,rj[0]:rj[1]+1,1])

            		norm = (pdf_S[i][j] * prodS * self.PS[scale][orient][0]) + (pdf_L[i][j] * prodL * self.PS[scale][orient][1])

            		BEC[scale][i][j][0] = (pdf_S[i][j] * prodS * self.PS[scale][orient][0]) / norm
            		BEC[scale][i][j][1] = (pdf_L[i][j] * prodL * self.PS[scale][orient][1]) / norm

            		BER[scale-1][ri[0]:ri[1]+1,rj[0]:rj[1]+1,:] = BEC[scale][i,j,:] /\
            										BEP[scale-1][ri[0]:ri[1]+1,rj[0]:rj[1]+1,:]

            		BEP[scale][i,j,:] = ((BEC[scale][i][j][0] * self.ES[scale][orient,0,:]) /self.PS[scale][orient][0] ) + \
										((BEC[scale][i][j][1] * self.ES[scale][orient,1,:])/self.PS[scale][orient][1])

        return (BEC, BEP, BER)


    def DOWN_Step(self, orient, BER):
        AL = strc.createArrayProb(self.Pyramid)
        scale = self.nLevels-1
        AL[scale][:,:,:] = 1

        for scale in range(self.nLevels-2, -1, -1):
            for i in range(AL[scale].shape[0]):
            	iP = i >> 1
            	for j in range(AL[scale].shape[1]):
                    jP = j >> 1
                    AL[scale][i, j, :] = ((AL[scale+1][iP][jP][0] * self.ES[scale, orient, :, 0] *\
                                        BER[scale][i][j][0]) + (AL[scale+1][iP][jP][1] * self.ES[scale, orient, :, 1] *\
                                        BER[scale][i][j][1])) /self.PS[scale][orient,:]

        return AL

    def computeProb(self, orient, BEC, BEP, BER, AL):

        for scale in range(self.nLevels):
            mult = AL[scale] * BEC[scale]
            self.P1[scale][:,:,orient,:] = mult

        for scale in range(self.nLevels-1):
            for i in range(self.P2[scale].shape[0]):
            	iP = i >> 1
            	for j in range(self.P2[scale].shape[1]):
                    jP = j >> 1
                    for f in range(2):
                        for p in range(2):
                            self.P2[scale][i,j,orient,f,p] = (BEC[scale][i,j,f] * self.ES[scale][orient][f][p] \
                                                            * AL[scale+1][iP,jP,p] * BER[scale][i,j,p]) / self.PS[scale][orient][f]


    def Maximization(self,orient):

        #atualizacao dos valores de probabilidade:
        for scale in range(self.nLevels):
            sumP = np.sum(self.P1[scale][:,:,orient],axis=(0,1))
            #numElem = math.pow(int(self.OriginalSize[0]/math.pow(2, scale)),2)
            #numElem = self.P1[scale][:,:,orient].size / sumP.size
            numElem = self.P1[scale].shape[0] * self.P1[scale].shape[1]
            self.PS[scale][orient,:] = sumP / numElem

        #sys.exit()

        for scale in range(self.nLevels-1):

            #numElem = math.pow(int(self.OriginalSize[0]/math.pow(2, scale)),2)
            numElem = self.P2[scale].shape[0] * self.P2[scale].shape[1]
            sumESIn = np.sum(self.P2[scale][:,:,orient],axis=(0,1))
            for i in range(2):
                for j in range(2):
                    self.ES[scale][orient][i][j] = sumESIn[i][j] / (numElem * self.PS[scale+1][orient,j])

        for scale in range(self.nLevels):
            WaveCoeffR = np.real(self.Pyramid[scale][:,:,orient])
            WaveCoeffI = np.imag(self.Pyramid[scale][:,:,orient])

            numElem = self.Pyramid[scale].shape[0] * self.Pyramid[scale].shape[1]

            tempS = numElem * self.PS[scale][orient][0]
            tempL = numElem * self.PS[scale][orient][1]

            tsis_re = np.sum(np.power(WaveCoeffR,2) * self.P1[scale][:,:,orient,0]) / tempS
            tsil_re = np.sum(np.power(WaveCoeffR,2) * self.P1[scale][:,:,orient,1]) / tempL

            tsis_im = np.sum(np.power(WaveCoeffI,2) * self.P1[scale][:,:,orient,0]) / tempS
            tsil_im = np.sum(np.power(WaveCoeffI,2) * self.P1[scale][:,:,orient,1]) / tempL


            self.SI[scale][orient][0] = np.sqrt(tsis_re * tsis_im)
            self.SI[scale][orient][1] = np.sqrt(tsil_re * tsil_im)

            # CoeffAbs = np.abs(self.Pyramid[scale][:,:,orient])
            # Coeff2 = np.power(CoeffAbs,2)

            # numElem = self.Pyramid[scale].shape[0] * self.Pyramid[scale].shape[1]
            # tempS = numElem * self.PS[scale][orient][0]
            # tempL = numElem * self.PS[scale][orient][1]

            # self.SI[scale,orient,0] = np.sum(Coeff2 * self.P1[scale][:,:,orient,0]) / tempS
            # self.SI[scale,orient,1] = np.sum(Coeff2 * self.P1[scale][:,:,orient,1]) / tempL


    def Convergency(self):

        perr = 0
        sierr = 0
        eserr = 0


        for scale in range(self.nLevels):
            tP = self.PS[scale] - self.PSO[scale]
            tP2 = np.power(tP,2)    
            perr += np.sum(tP2)

        totPoints = self.nLevels * self.nOrient

        tES = self.ES - self.ESO
        tES2 = np.power(tES,2)
        eserr = np.sum(tES2)

        tSI = self.SI - self.SIO
        tSI2 = np.power(tSI,2)
        sierr = np.sum(tSI2)

        perr /= totPoints
        sierr /= totPoints
        eserr /= totPoints

        print("ErrPS: ",perr)
        print("ErrSI: ",sierr)
        print("ErrES: ",eserr)

        return max(perr, sierr, eserr)

    def saveModel(self,path):
    	np.save(path+'/P1', self.P1)
    	np.save(path+'/P2', self.P2)

    	np.save(path+'/PS', self.PS)
    	np.save(path+'/ES', self.ES)
    	np.save(path+'/SI', self.SI)

    def LearnModel(self):
        noIter = 0
        err = 100

        self.createStructure()
        self.initializeParameters()

        it = 0
        small_error = sys.maxsize
        SMALL_NUMBER = 1E-2

        while(err > SMALL_NUMBER):
            self.storeParameters()

            print("Iter: ", it)

            for o in range(self.nOrient):
                #Expectation
                BEC, BEP, BER = self.UP_Step(o)
                AL = self.DOWN_Step(o, BER)
                #Maximization
                self.computeProb(o, BEC, BEP, BER, AL)
                self.Maximization(o)

            # f = open("saidas/ES"+str(it)+".txt","w")
            # f.write(str(self.ES))
            # f.close()

            # f = open("saidas/SI"+str(it)+".txt","w")
            # f.write(str(self.SI))
            # f.close()

            # f = open("saidas/Patualizado"+str(it)+".txt","w")
            # f.write(str(self.P))
            # f.close()

            # f = open("saidas/BEC"+str(it)+".txt","w")
            # f.write(str(BEC))
            # f.close()

            err = self.Convergency()
            it +=1

            # fSI = open("SI/SI"+str(it)+".txt","w")
            # fSI.write(str(self.SI))
            # fSI.close()


            if(err < small_error):
                small_error = err
                np.save('modeloSNUpdate/PS', self.PS)
                np.save('modeloSNUpdate/ES', self.ES)
                np.save('modeloSNUpdate/SI', self.SI)
                print("Saving parameters from resulting convergency error ", small_error)

            if(it == 150):
                SMALL_NUMBER = small_error
                print("New SMALL_NUMBER: ", SMALL_NUMBER)

        # f1 = open("saidas/P1.txt","w")
        # f1.write(str(self.P1))
        # f1.close()

        # f2 = open("saidas/P2.txt","w")
        # f2.write(str(self.P2))
        # f2.close()
