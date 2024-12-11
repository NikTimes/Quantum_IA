import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import h5py  # for efficient data storage

# Constants and parameters
x_min, x_max, N = -5, 5, 1024
x = np.linspace(x_min, x_max, N)
dx = x[1] - x[0]
dt = 0.01  # Time step
T = 2      # Total time
times = np.arange(0, T, dt)
psi = np.exp(-x**2).astype(np.complex128)  # Initial Gaussian wave packet

# Potential and Hamiltonian setup
V = 0.5 * x**2
H_diag = 1.0 / dx**2 + V
H_offdiag = -0.5 / dx**2 * np.ones(N - 1)
H = diags([H_diag, H_offdiag, H_offdiag], [0, -1, 1], format='csc', dtype=np.complex128)

# Crank-Nicolson matrices
I = eye(N, dtype=np.complex128, format='csc')  # Create a complex identity matrix in sparse format
A = I - 0.5j * dt * H
B = I + 0.5j * dt * H

# Data storage setup
with h5py.File('quantum_data.h5', 'w') as f:
    dset = f.create_dataset("data", (len(times), N), dtype='complex128')

    # Time evolution and data recording
    for idx, time in enumerate(times):
        b = B @ psi
        psi = spsolve(A, b)
        dset[idx, :] = psi  # store the wave function at this time step

    f.create_dataset("times", data=times)
    f.create_dataset("position", data=x)

