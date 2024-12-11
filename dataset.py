import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import torch
from torch.utils.data import Dataset

def generate_wavefunction_data(x_min=-5, x_max=5, N=1024, dt=0.01, T=2, omega=1.0):
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    times = np.arange(0, T, dt)
    psi = np.exp(-x**2).astype(np.complex128)  # Initial Gaussian wave packet

    # Potential and Hamiltonian setup with variable omega
    V = 0.5 * omega**2 * x**2
    H_diag = 1.0 / dx**2 + V
    H_offdiag = -0.5 / dx**2 * np.ones(N - 1)
    H = diags([H_diag, H_offdiag, H_offdiag], [0, -1, 1], format='csc', dtype=np.complex128)

    # Crank-Nicolson matrices
    I = eye(N, dtype=np.complex128, format='csc')
    A = I - 0.5j * dt * H
    B = I + 0.5j * dt * H

    psi_timesteps = []
    for time in times:
        b = B @ psi
        psi = spsolve(A, b)
        psi_timesteps.append(psi)

    return times, x, np.array(psi_timesteps)


class QuantumDataset(Dataset):
    def __init__(self, num_samples, x_min=-5, x_max=5, N=1024, dt=0.01, T=2, omega_range=(0.5, 2.0)):
        self.num_samples = num_samples
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.dt = dt
        self.T = T
        self.omega_range = omega_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly choose omega for each sample
        omega = np.random.uniform(*self.omega_range)
        times, x, psi_timesteps = generate_wavefunction_data(self.x_min, self.x_max, self.N, self.dt, self.T, omega)
        
        # Prepare data for neural network: flatten and combine real and imaginary parts
        psi_real = psi_timesteps.real.reshape(len(times), -1)
        psi_imag = psi_timesteps.imag.reshape(len(times), -1)
        data = np.concatenate([psi_real, psi_imag], axis=1)  # Shape: [time_steps, 2*N]
        
        # Convert to tensor (shape: [time_steps, 2*N])
        data_tensor = torch.from_numpy(data).float()
        return data_tensor


