from dataset import *
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from tqdm import tqdm
import wandb


class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Tanh(),
            nn.Linear(50, input_dim)
        )

    def forward(self, t, y):
        # y shape: [batch, input_dim]
        # Return dy/dt
        return self.net(y)
    

def train(model, train_loader, optimizer, criterion, dt, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                
                # batch shape: [batch_size, time_steps, features]
                # We predict from time steps [0..T-1] to [1..T]
                batch_input = batch[:, :-1, :]
                batch_target = batch[:, 1:, :]

                time_steps = batch_input.size(1)
                batch_t = torch.linspace(0, time_steps * dt, steps=time_steps)
                
                # odeint expects initial condition as shape [batch, features]
                # Make sure batch_input matches this shape
                # odeint will produce a result of shape [time_steps, batch, features]
                y0 = batch_input[:, 0, :]  # Shape: [batch_size, features]
                predicted = odeint(model, y0, batch_t)
                
                # predicted shape: [time_steps, batch_size, features]
                # batch_target shape: [batch_size, time_steps, features]
                # We need to transpose predicted to match target's shape: (batch, time_steps, features)
                predicted = predicted.permute(1, 0, 2)
                
                loss = criterion(predicted, batch_target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch": epoch+1, "loss": avg_loss})
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

if __name__ == "__main__":
    # Initialize Weights & Biases run
    wandb.init(project="quantum-ode-harmonic-oscillator", config={
        "num_samples": 200,
        "N": 1024,
        "dt": 0.01,
        "T": 2,
        "omega_range": (0.5, 1.5),
        "epochs": 5,
        "batch_size": 10,
        "learning_rate": 0.01
    })

    config = wandb.config
    
    dataset = QuantumDataset(num_samples=config.num_samples, N=config.N, dt=config.dt, T=config.T, omega_range=config.omega_range)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    input_dim = dataset[0].size(1)  # should be 2*N from dataset
    model = ODEFunc(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.HuberLoss(delta=1.0)

    train(model, loader, optimizer, criterion, dt=config.dt, epochs=config.epochs)
    torch.save(model.state_dict(), "model_checkpoint.pt")

    


