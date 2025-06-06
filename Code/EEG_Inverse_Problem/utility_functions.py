import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import lpmv, lpmn, lpn
from scipy.special import legendre

mu = 4 * math.pi * pow(10, -7)

def cartesian_to_polar(x, y, z):
    r = np.round(np.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)),4)
    theta = np.round(np.arccos(z / r),4)
    phi = np.round(np.arctan2(y, x),4)

    return r, theta, phi


#  ------------------------------- Convert spherical coordinates to Cartesian coordinates -------------------------------------
def Conv_coordinates(phi, theta, radius):

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z


#  ------------------------------- Convert Cartesian coordinates to spherical coordinates -------------------------------------

def cartesian_to_spherical(rq_x, rq_y, rq_z, radius):
    # Calculate φ
    phi = np.arctan2(rq_y, rq_x)
    
    # Calculate r
    r = np.sqrt(rq_x**2 + rq_y**2 + rq_z**2)
    
    # Calculate θ
    theta = np.arccos(rq_z / r)
    
    return theta, phi


