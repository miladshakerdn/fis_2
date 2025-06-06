import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank
from utility_functions import Conv_coordinates, cartesian_to_spherical

import math

# ----------------------- Loading Diapole and MEG_Lead_Field_1 -----------------------
#TODO: Load the diapole coordinates data from file
# data1 = ...
#TODO: Load the MEG lead field matrix
# data2 = ...
#TODO: Load the MEG measurement vector
# data3 = ...

#TODO: Extract the x, y, z coordinates from the daipole data
# rq_x = ...
# rq_y = ...
# rq_z = ...
# rq = ...

#TODO: Define the orientation vector for the diapole
# q_0 = ...

#TODO: Extract the lead field matrix and measurement vector
# G = ...
# B_r = ...

#TODO: Calculate and print the rank and shape of the lead field matrix
# Rank = ...
# print("Shape of G = ", G.shape)
# print("Rank of G = ", Rank)


# --------------------------- calculate Current_source_vector ------------------------
#TODO: Calculate the current source vector using the minimum norm solution
# q = ...
#TODO: Initialize an array to store the magnitude of each current source
# norm_q = ...

#TODO: Calculate the magnitude of each current source
# for i in range(0, 105):
    # ...
#TODO: Reshape the current source vector for easier analysis
# q1 = ...

#TODO: Print the estimated diapole orientation and related information
# print("q_0 vector = ", q1[104, :])
# print("diapole number = ", np.argmax(norm_q), "\nmaximum norm_q = ",
#       np.max(norm_q), "\nq_0 norm = ", norm_q[104])


#  --------------------------- Calculate the relative error --------------------------
#TODO: Reshape the current source vector if not already done
# q1 = ...

#TODO: Calculate the relative error between estimated and true diapole orientation
# relative_q0_error = ...
# print('The relative q0 error =', relative_q0_error)
#TODO: Calculate the relative error across all estimated sources
# relative_q_error = ...
# print('The relative q error =', relative_q_error)

# ------------------------------------------ Visiualize Current_source_vector -------------------------------------------------

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('Magnitude of Current Sources', c='r')

# Set size of each axis
ax.set_box_aspect([1, 1, 1])  # This will make the axes equally spaced
ax.set_xlim([-0.08, 0.08])
ax.set_ylim([-0.08, 0.08])
ax.set_zlim([-0.08, 0.08])

# Set the grid grading for each axis
ax.set_xticks(np.arange(-0.08, 0.08, 0.04))
ax.set_yticks(np.arange(-0.08, 0.08, 0.04))
ax.set_zticks(np.arange(-0.08, 0.08, 0.04))

q_min = np.min(q)
q_max = np.max(q)

# Plotting scatter with actual values
scatter = ax.scatter(rq_x, rq_y, rq_z, c=norm_q, cmap='viridis',
                     s=50, vmin=q_min, vmax=q_max)

# Adding color bar
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')


ax.quiver(rq[104, 0], rq[104, 1], rq[104, 2], q1[104, 0], q1[104, 1], q1[104, 2], color='r', length=0.04,
          normalize=True, arrow_length_ratio=0.5)
# -------------------------------------  Plot the hemisphere surface ------------------------------------------

radius = 0.07
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x_hemisphere = radius * np.outer(np.cos(u), np.sin(v))
y_hemisphere = radius * np.outer(np.sin(u), np.sin(v))
z_hemisphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_hemisphere, y_hemisphere,
                z_hemisphere, color='b', alpha=0.05)

ax.set_xlabel('X (m)', c='b')
ax.set_ylabel('Y (m)', c='b')
ax.set_zlabel('Z (m)', c='b')

plt.show()
