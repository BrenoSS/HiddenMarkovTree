import 	numpy   as np
import 	nibabel as nib
import 	dtcwt
from 	scipy import ndimage as nd
from 	skimage.feature import peak_local_max

filename = 'data/Template_T1_RAI.nii'

# *************************************************************************
base = filename.strip('.nii')
mask_filename   = base + '_mask.nii'
output_filename = base + '_Landmarks.nii'

# *************************************************************************
img = nib.load(filename)
img_mask = nib.load(mask_filename)

# *************************************************************************
# Visualmente, esses parametros funcionam bem 
# *************************************************************************
nScales = 2                           
alpha   = 1                                              
beta    = 0.25      
# *****************************************************     
energy_threshold = 0.2
# *****************************************************

# *************************************************************************
# Aparentmente, essa transformada produz melhores pontos 
# A outro opcao seria "trans = dtcwt.Transform3d()"
# *************************************************************************
# trans = dtcwt.Transform3d( biort='near_sym_b_bp', qshift='qshift_b_bp' )

# Z = trans.forward(img.get_data(), nlevels=nScales)
# Pyramid = Z.highpasses
# nOrientations = Pyramid[-1].shape[3];

# # Accumulator
# summ = np.zeros(Pyramid[-1].shape[:3])

# # Main part of the method
# for scale in range(nScales-1, -1, -1):
#     WaveCoeff = Pyramid[scale]
#     prod = np.ones(WaveCoeff.shape[:3])

#     for orient in range(0, nOrientations):
#     	# Compute the absolute of the complex coefficients
#         coeff_abs = np.abs(WaveCoeff[:,:,:,orient])

#         # Then, weight the coefficients according to the scale
#         prod = prod * pow(coeff_abs, beta)

#     summ = summ + pow(alpha, scale) * prod 
#     summ = nd.interpolation.zoom(summ, zoom=2, order=3)


summ = np.load("summ.npy")


# Perform detection of salient points only within the mask region 
mask = img_mask.get_data()
summ[ mask == 0 ] = 0

# Normalize energy map to the 0-1 range and then apply energy thredhold
summ = (summ - np.min(summ))/(np.max(summ) - np.min(summ))
summ[ summ < energy_threshold ] = 0

# Detect local maxima in the energy map
neighbor_26  = np.ones((3,3,3))
local_maxima = peak_local_max(summ, footprint=neighbor_26, indices=False, exclude_border=1)


points = np.argwhere(local_maxima == True)
print(points.shape[0])
print(points)

points2 = np.where(local_maxima == True)
print(points2)

pontos = [[1,1,1],[5,5,5]]

out = np.zeros(summ.shape)
out[points2] = 1

out_image = nib.Nifti1Image(out, img.get_affine())
nib.save(out_image, "teste.nii")

for x,y,z in points:
	print(x,y,z)


value = [73, 170, 162]

if value in points:
	print(value)

print(points.shape)

p = []
p.append(np.array(points[:,0]))
p.append(np.array(points[:,1]))
p.append(np.array(points[:,2]))

print(p)

out2 = np.zeros(summ.shape)
out2[p] = 1

out_image = nib.Nifti1Image(out2, img.get_affine())
nib.save(out_image, "teste2.nii")


out3 = np.zeros(summ.shape)
out3[np.array(points[:,0]),np.array(points[:,1]),np.array(points[:,2])] = 1

out_image = nib.Nifti1Image(out3, img.get_affine())
nib.save(out_image, "teste3.nii")




