import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank, pinv, norm
from utility_functions import Conv_coordinates, cartesian_to_spherical
import math

# ----------------------- Loading Diapole and EEG_Lead_Field_1 -----------------------
#TODO: Load the dipole coordinates data from file
data1 = np.load('Dataset/MEG/Diapole_coordinates_2.npz') # Coordinates are the same
#TODO: Load the EEG lead field matrix
data2 = np.load('Dataset/EEG/EEG_Lead_Field_1.npz')
#TODO: Load the EEG measurement vector
data3 = np.load('Dataset/EEG/EEG_Measurement_Vector.npz')

#TODO: Extract coordinates from the diapole data
rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']
rq = data1['rq']

#TODO: Define the orientation vector for the dipole (the "true" vector for error calculation)
q_0 = np.array([0, 0, 1])

#TODO: Extract the lead field matrix and measurement vector
L = data2['L']
V = data3['V']

#TODO: Calculate and print the rank and shape of the lead field matrix
Rank = matrix_rank(L)
print(f"Shape of L = {L.shape}")
print(f"Rank of L = {Rank}")

# --------------------------- Calculate Current Source Vector ------------------------
#TODO: Calculate the current source vector using the minimum norm solution
q = pinv(L) @ V

#TODO: Reshape the current source vector for easier analysis
q1 = q.reshape(105, 3)

#TODO: Initialize an array to store the magnitude of each current source
norm_q = np.zeros(105)

#TODO: Calculate the magnitude of each current source
for i in range(0, 105):
    norm_q[i] = norm(q1[i, :])

#TODO: Print the estimated diapole orientation and related information
print(f"Estimated q_0 vector = {q1[104, :]}")
print(f"Identified dipole number = {np.argmax(norm_q)}")
print(f"Maximum norm_q = {np.max(norm_q)}")
print(f"Norm of estimated q_0 = {norm_q[104]}")


# --------------------------- Calculate the Relative Error -------------------------
# Create the true total q vector for error calculation
q_true_total = np.zeros_like(q)
# The true source is the 105th one (index 104), with 3 components (qx,qy,qz)
start_index = 104 * 3
q_true_total[start_index : start_index + 3] = q_0

#TODO: Calculate relative q0 error
relative_q0_error = norm(q1[104, :] - q_0) / norm(q_0)
print(f"The relative q0 error = {relative_q0_error * 100:.2f}%")

#TODO: Calculate relative q error
relative_q_error = norm(q - q_true_total) / norm(q_true_total)
print(f"The total relative q error = {relative_q_error * 100:.2f}%")


# ------------------------------------------ Visiualize Current_source_vector -------------------------------------------------
# The visualization part is similar to the template provided
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('EEG Estimated Current Source Magnitude')

# Set size of each axis
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.08, 0.08, 0.04))
ax.set_yticks(np.arange(-0.08, 0.08, 0.04))
ax.set_zticks(np.arange(-0.08, 0.08, 0.04))

# Plotting scatter with actual values
scatter = ax.scatter(rq_x, rq_y, rq_z, c=norm_q, cmap='viridis', s=50)

# Adding color bar
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')

# Draw an arrow for the estimated vector at the true source location
ax.quiver(rq[104, 0], rq[104, 1], rq[104, 2], q1[104, 0], q1[104, 1], q1[104, 2], color='r', length=0.04,
          normalize=True, arrow_length_ratio=0.4)

# Plot the hemisphere surface
radius = 0.07
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='b', alpha=0.1) # Changed color for EEG

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()