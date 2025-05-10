import numpy as np


second_order_matrix = np.load("second_order_matrix.npy")
print(second_order_matrix.shape)

#https://stackoverflow.com/questions/45434989/numpy-difference-between-linalg-eig-and-linalg-eigh

# w, v = np.linalg.eigh(second_order_matrix)
# print(w.shape)


trace = np.trace(second_order_matrix,axis=(0,1,2))



# min_w = np.min(w,axis=-1)

# print("Max min: ", np.max(min_w))

# #min_w_norm = (min_w - np.min(min_w)) / (np.max(min_w) - np.min(min_w))

# out = np.zeros(min_w.shape)

# out[min_w > threshold] = 1

# print(np.argwhere(out==1).shape[0])