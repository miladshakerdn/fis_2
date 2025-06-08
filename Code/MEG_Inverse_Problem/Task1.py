import numpy as np
import math

# Announce the start of the calculation
print("Starting the calculation of the Lead Field Matrix (G)... This may take a while.")

# Define a physical constant (magnetic permeability of free space)
mu_0 = 4 * math.pi * 1e-7

# 1. Load the required input data
# Load the locations of 105 dipoles
dipole_data = np.load("../Dataset/MEG/Dipole_coordinates_2.npz")
rq = dipole_data['rq']  # 105x3 array of dipole locations

# Load the locations of 33 sensors
sensor_data = np.load("../Dataset/MEG/sensor_coordinates.npz")
r = np.vstack((sensor_data['x'], sensor_data['y'], sensor_data['z'])).T # 33x3 array of sensor locations

# Load the unit vectors of the 33 sensors
unit_vec_data = np.load("../Dataset/MEG/Unit_Vect_coordinates.npz")
er = np.vstack((unit_vec_data['ex'], unit_vec_data['ey'], unit_vec_data['ez'])).T # 33x3 array of unit vectors

# 2. Create an empty G matrix with the correct dimensions
num_sensors = r.shape[0]   # 33
num_dipoles = rq.shape[0]  # 105
G = np.zeros((num_sensors, num_dipoles * 3)) # Final dimensions: 33x315

# 3. Calculate the columns of the G matrix in a loop
# This loop iterates over each of the 105 dipole locations
for j in range(num_dipoles):
    rq_j = rq[j] # Location of the j-th dipole

    # For each dipole, we must calculate the effect of its three orthogonal components
    # qx = [1, 0, 0], qy = [0, 1, 0], qz = [0, 0, 1]
    for k in range(3):
        q_k = np.zeros(3)
        q_k[k] = 1.0 # Unit dipole in the k-th direction (x, y, or z)

        # Now, for this unit dipole, we calculate the field at all 33 sensors
        for i in range(num_sensors):
            r_i = r[i]   # Location of the i-th sensor
            er_i = er[i]  # Unit vector of the i-th sensor

            # Calculate the distance vector from the source to the sensor
            R_vec = r_i - rq_j
            R_norm = np.linalg.norm(R_vec)

            # Calculate the magnetic field vector using the Biot-Savart law for a dipole
            B_vec = (mu_0 / (4 * math.pi * R_norm**3)) * np.cross(q_k, R_vec)

            # Calculate the radial component of the field (what the MEG sensor measures)
            Br = np.dot(B_vec, er_i)

            # Store the calculated value in the appropriate column of the G matrix
            # The column corresponds to the k-th component of the j-th dipole
            G[i, j*3 + k] = Br

# 4. Save the final matrix
np.savez("../Dataset/MEG/MEG_Lead_Field_1.npz", G=G)

# Print a success message and the final matrix dimensions
print(f"File MEG_Lead_Field_1.npz was successfully generated.")
print(f"Dimensions of G matrix: {G.shape}")