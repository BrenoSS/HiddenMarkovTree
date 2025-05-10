import numpy as np

def createArrayProbOrient(tupla):
    # array de probabilidade com o índice de orientação
    #  return [np.zeros(list(arrays.shape)[:-1]+[2]) for arrays in tupla]
    return [np.zeros(list(arrays.shape)+[2]) for arrays in tupla]
    #return [np.zeros(arrays.shape+(2,)) for arrays in tupla]

def createArrayProb(tupla):
    # array de probabilidade sem o índice de orientação
    return [np.zeros(list(arrays.shape)[:-1]+[2]) for arrays in tupla]
    #return [np.zeros(list(arrays.shape)+[2]) for arrays in tupla]
    #return [np.zeros(arrays.shape+(2,)) for arrays in tupla]   

def createArrayTransf(tupla):
    # return [np.zeros(list(arrays.shape)[:-1]+[2,2]) for arrays in tupla]
    return [np.zeros(list(arrays.shape)+[2,2]) for arrays in tupla]

def createArray2D(nLevels, nOrient):
    return np.zeros((nLevels, nOrient, 2))

def createArrayMatrix(nLevels, nOrient):
    return np.zeros((nLevels, nOrient, 2, 2))

def createArrayShapeNoOrient(tupla):
    return [np.zeros((array.shape[0],array.shape[1])) for array in tupla]

def createArrayShape(tupla):
    return [np.zeros(array.shape) for array in tupla]