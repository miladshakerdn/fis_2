import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank, pinv, norm
from utility_functions import Conv_coordinates
import math

# ----------------------- Loading Dipole and EEG_Lead_Field_2 -----------------------
# Load dipole coordinates for plotting
data1 = np.load('Dataset/EEG/Dipole_coordinates_2.npz')
# Load the pre-computed lead field for the single known source
data2 = np.load('Dataset/EEG/EEG_Lead_Field_2.npz')
# Load the EEG measurement vector
data3 = np.load('Dataset/EEG/EEG_Measurement_Vector.npz')

# Extract coordinates for the source at index 104
rq = data1['rq']

# Define the true orientation vector for error calculation
q_0 = np.array([0, 0, 1])

# Extract the lead field matrix and measurement vector
L = data2['L']  # This is L_single, with shape (33, 3)
V = data3['V']

# Calculate and print the rank of the lead field matrix
Rank = matrix_rank(L)
print(f"Shape of L = {L.shape}")
print(f"Rank of L = {Rank}")  # Expected to be 3

# --------------------------- Calculate Current Source Vector ------------------------
# Calculate the current source vector using pseudoinverse
q = pinv(L) @ V

# Print shape and value of current source vector
print(f"Shape of estimated q = {q.shape}")
print(f"Estimated q vector (q_0) = {q}")

# Calculate magnitude of the current source
norm_q = norm(q)
print(f"Norm of estimated q = {norm_q}")

# --------------------------- Calculate the Relative Error -------------------------
# Calculate relative q0 error
relative_q0_error = norm(q - q_0) / norm(q_0)
print(f"The relative q0 error = {relative_q0_error * 100:.2f}%")

# In this case, the total error is the same as the error for q0
relative_q_error = relative_q0_error
print(f"The total relative q error = {relative_q_error * 100:.2f}%")

# -------------------------- Visiualize Current_source_vector -------------------------
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('EEG Estimated Current Source (Parametric)')

# Set axis properties
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])
ax.set_xticks(np.arange(-0.08, 0.08, 0.04))
ax.set_yticks(np.arange(-0.08, 0.08, 0.04))
ax.set_zticks(np.arange(-0.08, 0.08, 0.04))

# Plot the source location, colored by its estimated magnitude
scatter = ax.scatter(rq[104, 0], rq[104, 1], rq[104, 2], c=[norm_q], cmap='viridis', s=100)

# Add a color bar
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')

# Draw an arrow representing the estimated current vector
ax.quiver(rq[104, 0], rq[104, 1], rq[104, 2], q[0], q[1], q[2], color='r', length=0.04,
          normalize=True, arrow_length_ratio=0.4)

# Plot the hemisphere surface
radius = 0.07
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='b', alpha=0.1)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()