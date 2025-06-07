import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank, pinv, norm
from utility_functions import Conv_coordinates, cartesian_to_spherical
import math

# -------------------------- Loading Diapole and MEG_Lead_Field_2 ---------------------
#TODO: Load the diapole coordinates data from file
# Note: We only need the coordinates of the 105th dipole for plotting
data1 = np.load('Dataset/MEG/Diapole_coordinates_2.npz')

#TODO: Load the MEG lead field matrix for the single source
# This file is generated specifically for the known source location
data2 = np.load('Dataset/MEG/MEG_Lead_Field_2.npz')

#TODO: Load the MEG measurement vector
data3 = np.load('Dataset/MEG/MEG_Measurement_Vector.npz')

#TODO: Extract the x, y, z coordinates from the diapole data
# We only need the coordinates of the source at index 104 for the quiver plot
rq = data1['rq']

#TODO: Define the orientation vector for the diapole (the ground truth for error calculation)
q_0 = np.array([0, 0, 1])

#TODO: Extract the lead field matrix and measurement vector
G = data2['G'] # This is G_single, with expected shape (33, 3)
B_r = data3['B_r']

#TODO: Calculate and print the rank of the lead field matrix
Rank = matrix_rank(G)
print(f"Shape of G = {G.shape}")
print(f"Rank of G = {Rank}") # Expected to be 3 (Full Column Rank)


# --------------------------- calculate Current_source_vector ------------------------

#TODO: Calculate the current source vector using the pseudoinverse
q = pinv(G) @ B_r
print(f"Shape of estimated q = {q.shape}")
print(f"Estimated q vector = {q}")

#TODO: Calculate the magnitude of the current source
norm_q = norm(q)
print(f"Norm of estimated q = {norm_q}")


#  -------------------------- Calculate the relative error ---------------------------

#TODO: Calculate the relative error between estimated and true diapole orientation
relative_q0_error = norm(q - q_0) / norm(q_0)
print(f"The relative q0 error = {relative_q0_error * 100:.2f}%")

#TODO: Calculate the relative error (same calculation as above in this case)
# Since there is only one source, the total error is the same.
relative_q_error = relative_q0_error
print(f"The total relative q error = {relative_q_error * 100:.2f}%")


# ------------------------------------------ Visiualize Current_source_vector -------------------------------------------------

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('Estimated Current Source Vector (Parametric)',c='r')

# Set size of each axis
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.08, 0.08, 0.04))
ax.set_yticks(np.arange(-0.08, 0.08, 0.04))
ax.set_zticks(np.arange(-0.08, 0.08, 0.04))


# Plotting a single point for the source location
# The color of the point is based on the magnitude of the estimated vector
scatter = ax.scatter(rq[104, 0], rq[104, 1], rq[104, 2], c=[norm_q], cmap='viridis', s=100)

# Adding color bar
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')


# Plot the estimated vector as a quiver (arrow)
ax.quiver(rq[104, 0], rq[104, 1], rq[104, 2], q[0], q[1], q[2], color='r', length=0.04,
          normalize=True, arrow_length_ratio=0.4)
# -------------------------------------  Plot the hemisphere surface ------------------------------------------

radius = 0.07
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='r', alpha=0.1)

ax.set_xlabel('X (m)',c='b')
ax.set_ylabel('Y (m)',c='b')
ax.set_zlabel('Z (m)',c='b')

plt.show()