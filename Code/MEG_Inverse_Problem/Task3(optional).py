import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import pinv, norm
from utility_functions import cartesian_to_spherical
from scipy.interpolate import griddata
import math

# ------------------------------------------- Loading Dipole and MEG_Lead_Field_1 ------------------------------------------------
# This code assumes you have already run Task2 and have the results.
# We will reload the necessary data here for completeness.

# Load dipole locations
data1 = np.load('Dataset/MEG/Diapole_coordinates_2.npz')
rq = data1['rq']
rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']

# Load lead field and measurement vector to recalculate the estimated source
data2 = np.load('Dataset/MEG/MEG_Lead_Field_1.npz')
G = data2['G']
data3 = np.load('Dataset/MEG/MEG_Measurement_Vector.npz')
B_r = data3['B_r']

# -------------------------------------------- calculate Current_source_vector ---------------------------------------------------
# Recalculate the results from Task 2
q = pinv(G) @ B_r
q1 = q.reshape(105, 3)
norm_q = np.array([norm(q1[i, :]) for i in range(105)])


#  -------------------------------- Convert Cartesian coordinates to Spherical coordinates -------------------------------------
# We need the theta and phi for each dipole location to use as x and y axes for the surface plot
theta_dipoles = np.zeros(105)
phi_dipoles = np.zeros(105)
for i in range(105):
    # The function needs radius, but we only need theta and phi, so we can pass any radius
    theta_dipoles[i], phi_dipoles[i] = cartesian_to_spherical(rq_x[i], rq_y[i], rq_z[i], 0.07)

# Convert radians to degrees for plotting
theta_deg = np.rad2deg(theta_dipoles)
phi_deg = np.rad2deg(phi_dipoles)


# -------------------------------------------------- Plotting a 3D Meshgrid --------------------------------------------------
# A surface plot requires data on a regular grid. Our dipole data is scattered.
# We use scipy.interpolate.griddata to create a regular grid from our scattered data.

# 1. Create a regular grid for theta and phi
grid_theta_deg, grid_phi_deg = np.mgrid[min(theta_deg):max(theta_deg):100j, min(phi_deg):max(phi_deg):100j]

# 2. Interpolate the norm_q values (magnitudes) onto the new regular grid
grid_norm_q = griddata((theta_deg, phi_deg), norm_q, (grid_theta_deg, grid_phi_deg), method='cubic')


# 3. Create the 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Magnitude of Current Sources (Surface Plot)', c='r', fontsize=14)

# Plot the surface
surf = ax.plot_surface(grid_theta_deg, grid_phi_deg, grid_norm_q, cmap='viridis', edgecolor='none')

# Add labels and color bar
ax.set_xlabel('Theta (degree)', c='b', fontsize=12)
ax.set_ylabel('Phi (degree)', c='b', fontsize=12)
ax.set_zlabel('Magnitude (norm_q)', c='b', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5, label='Magnitude')

plt.show()

# ----------------------------------- Answering the questions using the data ------------------------------------

# Find the index of the dipole with the maximum norm
max_norm_index = np.argmax(norm_q)
# Get the value of the maximum norm
max_norm_value = np.max(norm_q)

# Get the coordinates of the identified source
theta_max_deg = theta_deg[max_norm_index]
phi_max_deg = phi_deg[max_norm_index]

# The actual source is the 105th dipole, which is at index 104
actual_source_index = 104

print(f"Maximum of the surface plot is located at index: {max_norm_index}")
print(f"Coordinates of maximum: Theta = {theta_max_deg:.2f} degrees, Phi = {phi_max_deg:.2f} degrees")
print(f"Is this the location of the actual current source? {'Yes' if max_norm_index == actual_source_index else 'No'}")
print(f"The size (value) of this maximum with respect to the color bar is: {max_norm_value}")