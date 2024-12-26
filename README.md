# Quantum Wavefunction Prediction

## Overview

This project focuses on simulating and predicting the time evolution of a quantum harmonic oscillator's wavefunction using a neural ODE (Ordinary Differential Equation) model. The implementation includes:

- **Data Generation**: Using Crank-Nicolson numerical integration to solve the Schrödinger equation for a quantum harmonic oscillator.
- **Model Training**: A Neural ODE model trained on generated data to predict the time evolution of the wavefunction.
- **Visualization**: Comparison of true vs. predicted wavefunction through animations.

## Key Features

1. **Wavefunction Simulation**:
   - Implemented via Crank-Nicolson integration for accuracy.
   - Includes customizable parameters like potential strength, spatial range, and time step size.

2. **Neural ODE Framework**:
   - Leverages `torchdiffeq` for solving differential equations.
   - Uses a simple feedforward neural network to predict wavefunction evolution.

3. **Visualization**:
   - A GIF animation comparing real and imaginary parts of the wavefunction between ground-truth data and model predictions.

## Results

The animation below shows the comparison of the real and imaginary parts of the true wavefunction (solid lines) and the predicted wavefunction (dashed lines) over time:

![Wavefunction Evolution](wavefunction_evolution.gif)


