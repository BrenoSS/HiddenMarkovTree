import numpy as np

def Rayleigh(w, sig):
    return float( ((w/sig) * np.exp(((-w)*w)/(2.0*sig))) + 1E-20 )

#https://stackoverflow.com/questions/7701429/efficient-evaluation-of-a-function-at-every-cell-of-a-numpy-array?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
Rayleigh = np.vectorize(Rayleigh)