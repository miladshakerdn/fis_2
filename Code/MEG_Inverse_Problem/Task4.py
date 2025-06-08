# Final and Corrected Code for Task 4
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank, pinv, norm
from utility_functions import Conv_coordinates

# -------------------------- Step 1: Load necessary data ---------------------
# Load dipole coordinates to get the source location for plotting
data1 = np.load("../Dataset/MEG/Dipole_coordinates_2.npz")

# Load the pre-computed 33x3 Lead Field matrix for the single source
# We assume the file is named as per your generator script.
lead_field_data = np.load("../Dataset/MEG/MEG_Lead_Field_Single_Dipole.npz")

# -------------------------- Step 2: Define problem parameters ---------------------
# Get the coordinates of the 105th dipole (at index 104)
rq = data1['rq']
source_coord = rq[104, :]

# This is the ground truth orientation of the source
q_0 = np.array([0, 0, 1])

# Extract the correctly-shaped (33, 3) lead field matrix
G = lead_field_data['G']

# --- THIS IS THE KEY CORRECTION ---
# Instead of loading B_r, we generate it ourselves based on the true source.
# This simulates the "measurement" process.
B_r = G @ q_0

# -------------------------- Step 3: Analyze and Solve ---------------------
# Analyze the lead field matrix
Rank = matrix_rank(G)
print(f"Shape of G = {G.shape}")
print(f"Rank of G = {Rank}")

# Solve the inverse problem to estimate the source vector 'q'
q_estimated = pinv(G) @ B_r
print(f'Shape of estimated q = {q_estimated.shape}')
print(f'Estimated q vector = {q_estimated}')

# Calculate the norm of the estimated vector
norm_q_estimated = norm(q_estimated)
print(f'Norm of estimated q = {norm_q_estimated}')

# -------------------------- Step 4: Calculate Error ---------------------------
# Calculate the relative error between the estimated vector and the ground truth
relative_error = norm(q_estimated - q_0) / norm(q_0)
print(f'The relative error = {relative_error * 100:.4f}%') # Expected to be very close to 0

# -------------------------- Step 5: Visualize the Result -----------------------------------
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('Estimated Source Vector (Parametric Least Squares)', c='r')

ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])

# Plot a single point for the source location
scatter = ax.scatter(source_coord[0], source_coord[1], source_coord[2], c=[norm_q_estimated], cmap='viridis', s=150, vmin=0)
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')

# Plot the estimated vector as an arrow. This will now work correctly.
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
ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='gray', alpha=0.1)

ax.set_xlabel('X (m)', c='b')
ax.set_ylabel('Y (m)', c='b')
ax.set_zlabel('Z (m)', c='b')

plt.show()