import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import pinv, norm
from utility_functions import cartesian_to_spherical
from scipy.interpolate import griddata
import math

# ----------------------- Step 1: Load All Necessary Data -----------------------
# Load dipole locations
data1 = np.load('../Dataset/MEG/Dipole_coordinates_2.npz')
rq = data1['rq']
rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']

# Load the FULL 33x315 EEG lead field matrix
data2 = np.load('../Dataset/EEG/EEG_Lead_Field_1.npz')
L = data2['L']

# ----------------------- Step 2: Define and Generate the Measurement Vector (V) -----------------------
# Define the true orientation of the actual source (the 105th dipole)
q_0 = np.array([0, 0, 1])

# --- THIS IS THE KEY CORRECTION ---
# Create the measurement vector V by simulating the signal from the true source.
# The true source is the 105th dipole, which corresponds to the last 3 columns of L.
L_true_source = L[:, -3:] # Shape: (33, 3)
V = L_true_source @ q_0   # V is the simulated potential vector, shape: (33,)

# ----------------------- Step 3: Calculate the MNE Solution -----------------------
# Recalculate the estimated current source and its norm for all 105 locations
q = pinv(L) @ V
q1 = q.reshape(105, 3)
norm_q = np.array([norm(q1[i, :]) for i in range(105)])


# ----------------------- Step 4: Prepare Data for Surface Plot -----------------------
# Get theta and phi for each dipole location to use as x and y axes for the surface plot
theta_dipoles = np.zeros(105)
phi_dipoles = np.zeros(105)
for i in range(105):
    theta_dipoles[i], phi_dipoles[i] = cartesian_to_spherical(rq_x[i], rq_y[i], rq_z[i], 0.07)

# Convert radians to degrees for plotting
theta_deg = np.rad2deg(theta_dipoles)
phi_deg = np.rad2deg(phi_dipoles)


# ----------------------- Step 5: Interpolate Data for a Smooth Plot -----------------------
# A surface plot requires data on a regular grid. Our dipole data is scattered.
# We use scipy.interpolate.griddata to create a regular grid from our scattered data.

# Create a regular grid for theta and phi
grid_theta_deg, grid_phi_deg = np.mgrid[min(theta_deg):max(theta_deg):100j, min(phi_deg):max(phi_deg):100j]

# Interpolate the norm_q values (magnitudes) onto the new regular grid
grid_norm_q = griddata((theta_deg, phi_deg), norm_q, (grid_theta_deg, grid_phi_deg), method='cubic')


# ----------------------- Step 6: Create the 3D Surface Plot -----------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('EEG Estimated Source Magnitude (Surface Plot)', c='r', fontsize=14)

# Plot the surface
surf = ax.plot_surface(grid_theta_deg, grid_phi_deg, grid_norm_q, cmap='viridis', edgecolor='none')

# Add labels and color bar
ax.set_xlabel('Theta (degree)', c='b', fontsize=12)
ax.set_ylabel('Phi (degree)', c='b', fontsize=12)
ax.set_zlabel('Magnitude (norm_q)', c='b', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5, label='Magnitude')

plt.show()

# ----------------------- Step 7: Print the Analysis of the Result -----------------------
# Find the index of the dipole with the maximum norm
max_norm_index = np.argmax(norm_q)
# Get the value of the maximum norm
max_norm_value = np.max(norm_q)

# Get the coordinates of the identified source
theta_max_deg = theta_deg[max_norm_index]
phi_max_deg = phi_deg[max_norm_index]

# The actual source is at index 104
actual_source_index = 104

print(f"Maximum of the surface plot is located at index: {max_norm_index}")
print(f"Coordinates of maximum: Theta = {theta_max_deg:.2f} degrees, Phi = {phi_max_deg:.2f} degrees")
print(f"Is this the location of the actual current source? {'Yes' if max_norm_index == actual_source_index else 'No'}")
print(f"The size (value) of this maximum is: {max_norm_value}")