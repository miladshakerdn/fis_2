import numpy as np
import math

print("Starting to compute the EEG Lead Field Matrix (L)... This may take a moment.")

# --- Define Constants ---
# Physical constant for conductivity (a typical value for the brain)
SIGMA = 0.3
# Number of sensors and dipoles
NUM_SENSORS = 33
NUM_DIPOLES = 105

# --- Load Geometric Information ---
# Load the 105 dipole locations
try:
    dipole_data = np.load("../Dataset/MEG/Dipole_coordinates_2.npz")
    rq = dipole_data['rq']  # Shape: (105, 3)
except FileNotFoundError:
    print("Error: 'Diapole_coordinates_2.npz' not found. Please generate it first.")
    exit()

# Load the 33 sensor locations
try:
    sensor_data = np.load("../Dataset/EEG/sensor_coordinates.npz")
    r = np.vstack((sensor_data['x'], sensor_data['y'], sensor_data['z'])).T # Shape: (33, 3)
except FileNotFoundError:
    print("Error: 'sensor_coordinates.npz' not found.")
    exit()

# --- Calculate the EEG Lead Field Matrix (L) ---
# Initialize an empty L matrix with the correct dimensions
L = np.zeros((NUM_SENSORS, NUM_DIPOLES * 3)) # Final shape: (33, 315)

# Loop over each of the 105 dipole locations
for j in range(NUM_DIPOLES):
    rq_j = rq[j] # Position vector of the j-th dipole

    # For each location, calculate the effect of its 3 orthogonal components
    for k in range(3):
        # Define a unit moment vector for the current basis direction (x, y, or z)
        q_basis_vector = np.zeros(3)
        q_basis_vector[k] = 1.0

        # Calculate the potential at all 33 sensors due to this unit moment
        for i in range(NUM_SENSORS):
            r_i = r[i] # Position vector of the i-th sensor

            # Calculate the vector from the source to the sensor
            R_vec = r_i - rq_j
            R_norm = np.linalg.norm(R_vec)

            # Avoid division by zero if a sensor is at the exact source location
            if R_norm == 0:
                L[i, j*3 + k] = 0
                continue

            # Calculate the potential V using the formula for an infinite homogeneous medium
            # V = (1 / (4*pi*sigma)) * (q . R) / ||R||^3
            V = (1 / (4 * math.pi * SIGMA)) * np.dot(q_basis_vector, R_vec) / (R_norm**3)

            # Store the calculated potential in the correct column of the L matrix
            L[i, j*3 + k] = V

# --- Save the Result ---
# Save the final L matrix to a .npz file
np.savez("../Dataset/EEG/EEG_Lead_Field_1.npz", L=L)

print(f"Successfully created 'EEG_Lead_Field_1.npz'.")
print(f"Shape of the generated L matrix: {L.shape}")