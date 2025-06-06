import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import matrix_rank
from utility_functions import Conv_coordinates
import math

# ----------------------- Loading Dipole and EEG_Lead_Field_1 -----------------------
#TODO: Load the dipole coordinates data from file
# data1 = ...
#TODO: Load the EEG lead field matrix
# data2 = ...
#TODO: Load the EEG measurement vector
# data3 = ...

#TODO: Extract coordinates from the dipole data
# rq_x = ...
# rq_y = ...
# rq_z = ...
# rq = ...

#TODO: Define the orientation vector for the dipole
# q_0 = ...

#TODO: Extract the lead field matrix and measurement vector
# L = ...
# V = dat...

#TODO: Calculate and print the rank of the lead field matrix
# Rank = ...
# print("Rank of L = ", Rank)

# --------------------------- Calculate Current Source Vector ------------------------
#TODO: Calculate the current source vector using pseudoinverse
# q = ...

#TODO: Print shape and value of current source vector
# print('shape of q = ', q.shape)
# print('q_0 = q =', q)

#TODO: Calculate magnitude of the current source
# norm_q = ...
# print('norm of q = ', norm_q)

# --------------------------- Calculate the Relative Error -------------------------
#TODO: Calculate relative q0 error
# relative_q0_error = ...
# print('The relative q0 error =', relative_q0_error)
#TODO: Calculate relative q error
# relative_q_error = ...
# print('The relative q error =', relative_q_error)


# ------------------------------------------ Visiualize Current_source_vector -------------------------------------------------

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
plt.title('Magnitude of Current Sources')       # change

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
scatter = ax.scatter(rq[104, 0], rq[104, 1], rq[104, 2], c=norm_q, cmap='viridis',
                     s=50, vmin=q_min, vmax=q_max)

# Adding color bar
cbar = plt.colorbar(scatter, pad=0.05)
cbar.set_label('norm_q')  # change


ax.quiver(rq[104, 0], rq[104, 1], rq[104, 2], q[0], q[1], q[2], color='r', length=0.03,
          normalize=True, arrow_length_ratio=0.5)
# -------------------------------------  Plot the hemisphere surface ------------------------------------------

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
