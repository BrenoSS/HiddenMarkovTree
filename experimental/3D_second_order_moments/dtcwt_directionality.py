#!/bin/python

"""
An example of the directional selectivity of 3D DT-CWT coefficients.

This example creates a 3D array holding an image of a sphere and performs the
3D DT-CWT transform on it. The locations of maxima (and their images about the
mid-point of the image) are determined for each complex coefficient at level 2.
These maxima points are then shown on a single plot to demonstrate the
directions in which the 3D DT-CWT transform is selective.

"""

# Import the libraries we need
import numpy as np
import dtcwt


def reducao_ao_1_q_r(angulo):
    if(angulo < (np.pi/2)):
        #1 quadrante
        return(angulo)
    elif((angulo >= (np.pi/2) ) and (angulo < np.pi)):
        #2 quadrante
        return(np.pi - angulo)
    elif((angulo >= np.pi ) and (angulo < ((3*np.pi)/2))):
        #3 quadrante
        return(angulo - np.pi)
    elif((angulo >= ((3*np.pi)/2) ) and (angulo < (2*np.pi))):
        return((2*np.pi) - angulo)


def reducao_ao_1_q_d(angulo):
    if(angulo < 90):
        #1 quadrante
        return(angulo)
    elif((angulo >= 90 ) and (angulo < 180)):
        #2 quadrante
        return(180 - angulo)
    elif((angulo >= 180 ) and (angulo < 270)):
        #3 quadrante
        return(angulo - 180)
    elif((angulo >= 270 ) and (angulo < 360)):
        return(360 - angulo)



def angle_between_vectors_azimuth_r(v1, v2):
    cos = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos)

    # finding the analogous angle in the first quadrant

    angulo = reducao_ao_1_q_r(angle)

    if((v1[0] > 0) and (v1[1] > 0)):
        #1 quadrante
        angulo = angulo
    elif((v1[0] < 0) and (v1[1] > 0)):
        #2 quadrante
        angulo = np.pi - angulo
    elif((v1[0] < 0) and (v1[1] < 0)):
        #3 quadrante
        angulo = angulo + np.pi
    elif((v1[0] > 0) and (v1[1] < 0)):
        angulo = (2*np.pi) - angulo
    return(angulo)


def angle_between_vectors_elevation_r(v1, v2):
    cos = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos)

    angulo = reducao_ao_1_q_r(angle)

    if((v1[1] > 0) and (v1[2] > 0)):
        #1 quadrante
        angulo = angulo
    elif((v1[1] < 0) and (v1[2] > 0)):
        #2 quadrante
        angulo = np.pi - angulo
    elif((v1[1] < 0) and (v1[2] < 0)):
        #3 quadrante
        angulo = angulo + np.pi
    elif((v1[1] > 0) and (v1[2] < 0)):
        angulo = (2 * np.pi) - angulo
    return(angulo)


def angle_between_vectors_azimuth_d(v1, v2):
    cos = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cos))

    # finding the analogous angle in the first quadrant

    angulo = reducao_ao_1_q_d(angle)

    # if((v1[0] > 0) and (v1[1] < 0)):
    #     angle = 360 - angle
    # elif((v1[0] < 0) and (v1[1] < 0)):
    #     angle = 180 + (180 - angle)
    if((v1[0] > 0) and (v1[1] > 0)):
        #1 quadrante
        angulo = angulo
    elif((v1[0] < 0) and (v1[1] > 0)):
        #2 quadrante
        angulo = 180 - angulo
    elif((v1[0] < 0) and (v1[1] < 0)):
        #3 quadrante
        angulo = angulo + 180
    elif((v1[0] > 0) and (v1[1] < 0)):
        angulo = 360 - angulo
    return(angulo)


def angle_between_vectors_elevation_d(v1, v2):
    cos = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cos))

    angulo = reducao_ao_1_q_d(angle)

    # if((v1[1] > 0) and (v1[2] < 0)):
    #     angle = 360 - angle
    # elif((v1[1] < 0) and (v1[2] < 0)):
    #     angle = 180 + (180 - angle)
    if((v1[1] > 0) and (v1[2] > 0)):
        #1 quadrante
        angulo = angulo
    elif((v1[1] < 0) and (v1[2] > 0)):
        #2 quadrante
        angulo = 180 - angulo
    elif((v1[1] < 0) and (v1[2] < 0)):
        #3 quadrante
        angulo = angulo + 180
    elif((v1[1] > 0) and (v1[2] < 0)):
        angulo = 360 - angulo
    return(angulo)



def angle_infos():
    GRID_SIZE = 128
    SPHERE_RAD = 0.33 * GRID_SIZE

    # Compute an image of the sphere
    grid = np.arange(-(GRID_SIZE>>1), GRID_SIZE>>1)
    X, Y, Z = np.meshgrid(grid, grid, grid)
    r = np.sqrt(X*X + Y*Y + Z*Z)
    sphere = 0.5 + np.clip(SPHERE_RAD-r, -0.5, 0.5)

    # Specify number of levels and wavelet family to use
    nlevels = 2

    # Form the DT-CWT of the sphere
    trans = dtcwt.Transform3d(biort='near_sym_b_bp', qshift='qshift_b_bp')
    Z = trans.forward(sphere, nlevels=nlevels)
    Yh = Z.highpasses
    
    locs = []
    scale = 1.1

    normal = np.array([0, 0, -1]) #plano do ângulo azimuth
    normal_2 = np.array([1, 0, 0]) #plano do ângulo de elevação

    angle_a = []
    angle_e = []

    for idx in range(Yh[-1].shape[3]):
        Z = Yh[-1][:,:,:,idx]
        C = np.abs(Z)
        max_loc = np.asarray(np.unravel_index(np.argmax(C), C.shape)) - np.asarray(C.shape)*0.5
        max_loc /= np.sqrt(np.sum(max_loc * max_loc))
        locs.append(max_loc)
        scalar_distance = normal.dot(max_loc)
        new_p = max_loc - (normal.dot(max_loc) * normal)
        new_p_2 = max_loc - (normal_2.dot(max_loc) * normal_2)

        angle_a.append(angle_between_vectors_azimuth_r(new_p,[1, 0, 0]))
        angle_e.append(angle_between_vectors_elevation_r(new_p_2,[0, 1, 0]))

        #print("Orientation " + str(idx) + " - Azimuth: " + str(angle_a) + ", Elevation: " + str(angle_e))
    return(angle_a, angle_e)

                        



