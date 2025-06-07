import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank, pinv, norm

# ----------------------- Loading Diapole and MEG_Lead_Field_1 -----------------------
#TODO: Load the diapole coordinates data from file
data1 = np.load('Dataset/MEG/Diapole_coordinates_2.npz')
#TODO: Load the MEG lead field matrix
data2 = np.load('Dataset/MEG/MEG_Lead_Field_1.npz')
#TODO: Load the MEG measurement vector
data3 = np.load('Dataset/MEG/MEG_Measurement_Vector.npz')

#TODO: Extract the x, y, z coordinates from the daipole data
rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']
rq = data1['rq']

#TODO: Define the orientation vector for the dipole
q_0 = np.array([0, 0, 1])  # The true orientation of the 105th source

#TODO: Extract the lead field matrix and measurement vector
G = data2['G']
B_r = data3['B_r']

#TODO: Calculate and print the rank and shape of the lead field matrix
Rank = matrix_rank(G)
print("Shape of G = ", G.shape)
print("Rank of G = ", Rank)

# --------------------------- calculate Current_source_vector ------------------------
#TODO: Calculate the current source vector using the minimum norm solution
q = pinv(G) @ B_r

#TODO: Reshape the current source vector for easier analysis
q1 = q.reshape(105, 3)

#TODO: Initialize an array to store the magnitude of each current source
norm_q = np.zeros(105)

#TODO: Calculate the magnitude of each current source
for i in range(0, 105):
    norm_q[i] = norm(q1[i, :])

#TODO: Print the estimated diapole orientation and related information
print("Estimated q_0 vector = ", q1[104, :])
print("Identified dipole number = ", np.argmax(norm_q), "\nMaximum norm_q = ",
      np.max(norm_q), "\nNorm of estimated q_0 = ", norm_q[104])

#  --------------------------- Calculate the relative error --------------------------
# Create the true q vector
q_true_total = np.zeros((105, 3))
q_true_total[104, :] = q_0
q_true_total = q_true_total.flatten()

#TODO: Calculate the relative error between estimated and true diapole orientation
relative_q0_error = norm(q1[104, :] - q_0) / norm(q_0)
print('The relative q0 error =', relative_q0_error)

#TODO: Calculate the relative error across all estimated sources
relative_q_error = norm(q - q_true_total) / norm(q_true_total)
print('The relative q error =', relative_q_error)

# --- Visualization code as provided in the problem ---