import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import pinv, norm
from utility_functions import cartesian_to_spherical
from scipy.interpolate import griddata
import math

# ------------------------------------------- Loading Dipole and EEG_Lead_Field_1 ------------------------------------------------
# This code block re-calculates the necessary results from Task 5.

# Load dipole coordinates, EEG lead field, and EEG measurements
data1 = np.load('Dataset/MEG/Diapole_coordinates_2.npz')
data2 = np.load('Dataset/EEG/EEG_Lead_Field_1.npz')
data3 = np.load('Dataset/EEG/EEG_Measurement_Vector.npz')

rq = data1['rq']
L = data2['L']
V = data3['V']

# -------------------------------------------- calculate Current_source_vector ---------------------------------------------------
# Recalculate the estimated current source and its norm for all 105 locations
q = pinv(L) @ V
q1 = q.reshape(105, 3)
norm_q = np.array([norm(q1[i, :]) for i in range(105)])

#  -------------------------------- Convert Cartesian to Spherical coordinates -------------------------------------
# Get theta and phi for each dipole to use as axes for the surface plot
theta_dipoles = np.zeros(105)
phi_dipoles = np.zeros(105)
for i in range(105):
    theta_dipoles[i], phi_dipoles[i] = cartesian_to_spherical(rq[i,0], rq[i,1], rq[i,2], 0.07)

# Convert radians to degrees for plotting
theta_deg = np.rad2deg(theta_dipoles)
phi_deg = np.rad2deg(phi_dipoles)

# -------------------------------------------------- Plotting a 3D Meshgrid --------------------------------------------------
# Interpolate scattered data onto a regular grid for smooth surface plotting
grid_theta_deg, grid_phi_deg = np.mgrid[min(theta_deg):max(theta_deg):100j, min(phi_deg):max(phi_deg):100j]
grid_norm_q = griddata((theta_deg, phi_deg), norm_q, (grid_theta_deg, grid_phi_deg), method='cubic')

# Create the 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('EEG Estimated Source Magnitude (Surface Plot)', c='r', fontsize=14)

# Plot the surface
surf = ax.plot_surface(grid_theta_deg, grid_phi_deg, grid_norm_q, cmap='viridis', edgecolor='none')

# Add labels and a color bar
ax.set_xlabel('Theta (degree)', c='b', fontsize=12)
ax.set_ylabel('Phi (degree)', c='b', fontsize=12)
ax.set_zlabel('Magnitude (norm_q)', c='b', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5, label='Magnitude')

plt.show()

# ----------------------------------- Programmatic analysis to answer the questions ------------------------------------

# Find the index and value of the maximum estimated norm
max_norm_index = np.argmax(norm_q)
max_norm_value = np.max(norm_q)

# Get the spherical coordinates of the identified source
theta_max_deg = theta_deg[max_norm_index]
phi_max_deg = phi_deg[max_norm_index]

# The actual source is at index 104
actual_source_index = 104

print(f"The maximum of the surface plot is programmatically found at dipole index: {max_norm_index}")
print(f"Coordinates of this maximum: Theta = {theta_max_deg:.2f} degrees, Phi = {phi_max_deg:.2f} degrees")
print(f"Is this the location of the actual source (index 104)? {'Yes' if max_norm_index == actual_source_index else 'No'}")
print(f"The size (value) of this maximum is: {max_norm_value}")