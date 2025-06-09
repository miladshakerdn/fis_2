import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank, pinv, norm
from utility_functions import Conv_coordinates, cartesian_to_spherical
import math

# ----------------------- Step 1: Load All Necessary Data -----------------------
# Load the 105 dipole locations (same for EEG and MEG)
data1 = np.load("../Dataset/MEG/Dipole_coordinates_2.npz")
# Load the FULL 33x315 EEG lead field matrix
data2 = np.load("../Dataset/EEG/EEG_Lead_Field_1.npz")

# ----------------------- Step 2: Define and Generate the Measurement Vector (V) -----------------------
# Extract all dipole coordinates
rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']
rq = data1['rq']

# Define the true orientation of the actual source (the 105th dipole)
q_0 = np.array([0, 0, 1])

# Extract the full EEG lead field matrix L
L = data2['L']

# --- KEY LOGIC ---
# We create the measurement vector V by simulating the signal from the true source.
# The true source is the 105th dipole, which corresponds to the last 3 columns of L.
L_true_source = L[:, -3:] # Shape will be (33, 3)
V = L_true_source @ q_0   # V is the simulated potential vector, shape will be (33,)

# ----------------------- Step 3: Analyze and Solve the Inverse Problem -----------------------
# Analyze the full lead field matrix
Rank = matrix_rank(L)
print(f"Shape of L = {L.shape}")
print(f"Rank of L = {Rank}")

# Solve for the source distribution 'q' using the Minimum Norm method
q = pinv(L) @ V

# Reshape the 315-element vector 'q' into a 105x3 matrix for analysis
q1 = q.reshape(105, 3)

# ----------------------- Step 4: Analyze the Results -----------------------
# Calculate the magnitude of each of the 105 estimated source vectors
norm_q = np.zeros(105)
for i in range(105):
    norm_q[i] = norm(q1[i, :])

# Print the results
print(f"Estimated q_0 vector (at index 104) = {q1[104, :]}")
print(f"Identified dipole number (index) = {np.argmax(norm_q)}")
print(f"Maximum norm_q = {np.max(norm_q)}")
print(f"Norm of estimated q_0 = {norm_q[104]}")

# ----------------------- Step 5: Calculate Error --------------------------
# Create the true source distribution vector for error comparison
q_true_total = np.zeros_like(q)
q_true_total[-3:] = q_0 # The only non-zero activity is the last 3 elements

# Calculate relative error for the orientation at the true source location
relative_q0_error = norm(q1[104, :] - q_0) / norm(q_0)
print(f'The relative q0 error = {relative_q0_error * 100:.2f}%')

# Calculate relative error for the entire source distribution
relative_q_error = norm(q - q_true_total) / norm(q_true_total)
print(f'The total relative q error = {relative_q_error * 100:.2f}%')

# ----------------------- Step 6: Visualize the Result -----------------------------------
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('EEG Estimated Current Source Magnitude', c='r')

ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])
ax.set_xticks(np.arange(-0.08, 0.08, 0.04))
ax.set_yticks(np.arange(-0.08, 0.08, 0.04))
ax.set_zticks(np.arange(-0.08, 0.08, 0.04))

scatter = ax.scatter(rq_x, rq_y, rq_z, c=norm_q, cmap='viridis', s=50)
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')

ax.quiver(rq[104, 0], rq[104, 1], rq[104, 2], q1[104, 0], q1[104, 1], q1[104, 2], color='r', length=0.04,
          normalize=True, arrow_length_ratio=0.5)

radius = 0.07
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='r', alpha=0.1)

ax.set_xlabel('X (m)', c='b')
ax.set_ylabel('Y (m)', c='b')
ax.set_zlabel('Z (m)', c='b')

plt.show()