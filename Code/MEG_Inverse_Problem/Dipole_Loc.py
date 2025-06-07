import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utility_functions import Conv_coordinates

# ---------------------  Calculate Coordinates of Diapole  ------------------
#TODO: Load the dipole coordinates from the dataset file
data1 = np.load("Dataset/MEG/Dipole_coordinates_1.npz")  # Loading Dipole

#TODO: Define the radius of the hemisphere
radius = 0.07  # radius of the hemisphere

#TODO: Extract the x, y, z coordinates from the loaded data
rq_x = data1['x']
rq_y = data1['y']
rq_z = data1['z']

#TODO: Concatenate the coordinates into a single array
rq = np.vstack((rq_x, rq_y, rq_z)).T

#TODO: Define the angles for the new dipole in radians
theta = np.deg2rad(45)
phi = np.deg2rad(45)
radius = 0.07

#TODO: Convert spherical coordinates to Cartesian coordinates
x_0, y_0, z_0 = Conv_coordinates(phi, theta, radius)

#TODO: Create the position vector for the new dipole
rq_0 = np.array([x_0, y_0, z_0])

#TODO: Define the orientation vector for the new dipole
q_0 = np.array([0, 0, 1]) # Assuming orientation along z-axis for visualization

#TODO: Append the new dipole position to the existing coordinates
rq = np.vstack([rq, rq_0])

#TODO: Save the updated dipole coordinates to a new file
np.savez("Dataset/MEG/Diapole_coordinates_2.npz", x=rq[:,0], y=rq[:,1], z=rq[:,2], rq=rq)


# -------------------------------------  Visiualize Diapole  ------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Random Non Reapeted Diapoles')

# Set size of each axis
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.08, 0.08, 0.04))
ax.set_yticks(np.arange(-0.08, 0.08, 0.04))
ax.set_zticks(np.arange(-0.08, 0.08, 0.04))

ax.set_box_aspect([1, 1, 1])

# Scatter plot for MEG sensors with numbers
for i, (xi, yi, zi) in enumerate(zip(rq[:,0], rq[:,1], rq[:,2])):
    ax.scatter(xi, yi, zi, color='b')
    if(i == 104):
        ax.text(xi, yi, zi, f'{i+1}', color='black', fontsize=9)

ax.quiver(rq_0[0], rq_0[1], rq_0[2], q_0[0], q_0[1], q_0[2], color='r', length=0.03,
          normalize=True, arrow_length_ratio=0.5)

# -------------------------------------  Plot the hemisphere surface ------------------------------------------

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_hemisphere, y_hemisphere, z_hemisphere, color='r', alpha=0.1) # Changed alpha for better visibility

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

plt.show()