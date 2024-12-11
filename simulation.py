from dataset import *
from Quantum_ODE import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchdiffeq import odeint
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --------------------
# Configuration
# --------------------
x_min = -5
x_max = 5
N = 1024
dt = 0.01
T = 2.0
omega = 1.0
time_steps = int(T / dt)
input_dim = 2 * N  # Real and imaginary parts

# --------------------
# Generate reference data (Crank-Nicolson solution)
# --------------------
times, x, psi_timesteps = generate_wavefunction_data(x_min, x_max, N, dt, T, omega)
# psi_timesteps shape: [time_steps, N], complex
psi_real_true = psi_timesteps.real
psi_imag_true = psi_timesteps.imag

# Prepare initial condition for ODE model
# The ODE model expects a shape [batch, features]
# We'll treat a single simulation as batch=1
data = np.concatenate([psi_real_true[0, :].reshape(1, -1),
                       psi_imag_true[0, :].reshape(1, -1)], axis=1)
y0 = torch.from_numpy(data).float()  # Shape: [1, 2*N]

time_tensor = torch.linspace(0, (time_steps - 1) * dt, steps=time_steps)

# --------------------
# Load the trained model
# --------------------
model = ODEFunc(input_dim=input_dim)
model.load_state_dict(torch.load("model_checkpoint.pt", map_location=torch.device('cpu')))
checkpoint = torch.load("model_checkpoint.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# --------------------
# Predict solution with ODE model
# --------------------
# odeint will produce a solution of shape [time_steps, batch, features]
with torch.no_grad():
    predicted = odeint(model, y0, time_tensor)  # shape: [time_steps, 1, 2*N]
predicted = predicted.squeeze(1).numpy()  # shape: [time_steps, 2*N]

psi_real_pred = predicted[:, :N]
psi_imag_pred = predicted[:, N:]

# --------------------
# Create an animation comparing true vs predicted wavefunction
# We'll create two subplots: one for the real part and one for the imaginary part
# Top row: Real part (True and Pred)
# Bottom row: Imag part (True and Pred)
# --------------------
fig, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(10, 8))

line_real_true, = ax_real.plot([], [], 'b-', label='Real True', alpha=0.7)
line_real_pred, = ax_real.plot([], [], 'r--', label='Real Pred', alpha=0.7)
ax_real.set_xlim(x_min, x_max)
ax_real.set_ylim(
    min(psi_real_true.min(), psi_real_pred.min()) * 1.1,
    max(psi_real_true.max(), psi_real_pred.max()) * 1.1
)
ax_real.set_xlabel('x')
ax_real.set_ylabel('Re(psi)')
ax_real.legend()

line_imag_true, = ax_imag.plot([], [], 'b-', label='Imag True', alpha=0.7)
line_imag_pred, = ax_imag.plot([], [], 'r--', label='Imag Pred', alpha=0.7)
ax_imag.set_xlim(x_min, x_max)
ax_imag.set_ylim(
    min(psi_imag_true.min(), psi_imag_pred.min()) * 1.1,
    max(psi_imag_true.max(), psi_imag_pred.max()) * 1.1
)
ax_imag.set_xlabel('x')
ax_imag.set_ylabel('Im(psi)')
ax_imag.legend()

time_text = ax_real.text(0.02, 0.95, '', transform=ax_real.transAxes)

def init():
    line_real_true.set_data([], [])
    line_real_pred.set_data([], [])
    line_imag_true.set_data([], [])
    line_imag_pred.set_data([], [])
    time_text.set_text('')
    return line_real_true, line_real_pred, line_imag_true, line_imag_pred, time_text

def update(frame):
    # frame: index of the time step
    line_real_true.set_data(x, psi_real_true[frame, :])
    line_real_pred.set_data(x, psi_real_pred[frame, :])
    line_imag_true.set_data(x, psi_imag_true[frame, :])
    line_imag_pred.set_data(x, psi_imag_pred[frame, :])
    time_text.set_text(f'Time = {times[frame]:.2f}')
    return line_real_true, line_real_pred, line_imag_true, line_imag_pred, time_text

ani = animation.FuncAnimation(fig, update, frames=time_steps, init_func=init, interval=50, blit=True)
ani.save("wavefunction_evolution.gif", writer="pillow", fps=20)

plt.tight_layout()
plt.show()
