# Final and Corrected Code for Task 7: EEG Parametric Least Squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank, pinv, norm
from utility_functions import Conv_coordinates
import math
import sys

# ----------------------- Step 1: Load Geometric Data and Define Constants -----------------------
# Load dipole coordinates to get the location of the single known source
try:
    data1 = np.load("../Dataset/MEG/Dipole_coordinates_2.npz")
    rq = data1['rq']
    source_coord = rq[104, :] # The 105th dipole is our source
except FileNotFoundError:
    print("Error: 'Dipole_coordinates_2.npz' not found. Please generate it first.")
    sys.exit()

# Load sensor coordinates
try:
    data2 = np.load("../Dataset/MEG/sensor_coordinates.npz")
    r = np.vstack((data2['x'], data2['y'], data2['z'])).T
except FileNotFoundError:
    print("Error: 'sensor_coordinates.npz' not found.")
    sys.exit()

# Define physical constants
SIGMA = 0.3 # Brain/scalp conductivity
NUM_SENSORS = 33

# ----------------------- Step 2: Generate the Parametric EEG Lead Field (L) -----------------------
# For this task, we need a 33x3 matrix where each column represents the potential
# from a unit dipole in the x, y, or z direction at the known source location.
L_single = np.zeros((NUM_SENSORS, 3))

# Loop over the three basis directions (k=0 for x, k=1 for y, k=2 for z)
for k in range(3):
    q_basis_vector = np.zeros(3)
    q_basis_vector[k] = 1.0

    # Calculate the potential at all 33 sensors for this unit moment
    for i in range(NUM_SENSORS):
        R_vec = r[i] - source_coord
        R_norm = np.linalg.norm(R_vec)
        if R_norm == 0: continue

        # Calculate potential V using the formula for an infinite homogeneous medium
        V_ik = (1 / (4 * math.pi * SIGMA)) * np.dot(q_basis_vector, R_vec) / (R_norm**3)
        L_single[i, k] = V_ik

# -------------------------- Step 3: Define Problem and Solve ---------------------
# Define the ground truth orientation of the source
q_0 = np.array([0, 0, 1])

# Generate the measurement vector V by simulating the signal from the true source
V = L_single @ q_0

# Analyze the generated lead field matrix
Rank = matrix_rank(L_single)
print(f"Shape of L = {L_single.shape}")
print(f"Rank of L = {Rank}")

# Solve the inverse problem to estimate the source vector 'q'
q_estimated = pinv(L_single) @ V
print(f'Shape of estimated q = {q_estimated.shape}')
print(f'Estimated q vector = {q_estimated}')

# --------------------------- Step 4: Calculate Error and Analyze -------------------------
# Calculate the magnitude of the estimated vector
norm_q = norm(q_estimated)
print(f'Norm of estimated q = {norm_q}')

# Calculate the relative error
relative_error = norm(q_estimated - q_0) / norm(q_0)
print(f'The relative error = {relative_error * 100:.4f}%')

# ------------------------------------------ Step 5: Visualize Result -------------------------------------------------
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('EEG Estimated Source Vector (Parametric)', c='r')

ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])
ax.set_xticks(np.arange(-0.08, 0.09, 0.04))
ax.set_yticks(np.arange(-0.08, 0.09, 0.04))
ax.set_zticks(np.arange(-0.08, 0.09, 0.04))

# Plot the source location, colored by its estimated magnitude
scatter = ax.scatter(source_coord[0], source_coord[1], source_coord[2], c=[norm_q], cmap='viridis', s=150)
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')

# Plot the estimated vector as an arrow
ax.quiver(source_coord[0], source_coord[1], source_coord[2],
          q_estimated[0], q_estimated[1], q_estimated[2],
          color='r', length=0.04, normalize=True, arrow_length_ratio=0.5)

# Plot the hemisphere surface
radius = 0.07
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='r', alpha=0.1)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()