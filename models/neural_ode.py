# -*- coding: utf-8 -*-
"""
Neural ODE Architecture — Project Confluence
============================================

Implements a continuous-time sequence-to-sequence model using PyTorch
and torchdiffeq. This acts as an alternative to the Bayesian MCMC fitter,
learning the nonlinear 15D vector field (metabolic + immune + stromal)
directly from sparse clinical time-series data.

Usage:
    from models.neural_ode import ComplexityNeuralODE, train_neural_ode
    model = ComplexityNeuralODE()
    # model.forward(t, z0)
"""

from typing import Tuple, Dict, Optional, List

# Try importing torch and torchdiffeq
try:
    import torch
    import torch.nn as nn
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    
    # Create dummy classes to prevent NameError syntax crashes
    class DummyModule:
        pass
    class nn:
        Module = DummyModule


class ODEF(nn.Module):
    """
    The neural derivative mapping z(t) -> dz/dt.
    Represents the 15D biological state equations without hardcoding 
    the Michaelis-Menten kinetics.
    """
    def __init__(self, state_dim: int = 15, hidden_dim: int = 64):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Initialize with small weights to keep initial dynamics near 0
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        # The Neural ODE requires the signature f(t, z) -> dz/dt
        # z may be [batch, state_dim] or [state_dim]
        out = self.net(z)
        
        # Soft-clamp extremely large gradients to prevent ODE solver from dying
        # biological states do not change at infinite velocity
        return torch.clamp(out, min=-10.0, max=10.0)


class TrajectoryEncoder(nn.Module):
    """
    Encodes sparse patient time-series data into the latent initial state (z0).
    Using a GRU to process (timestamp, observations).
    """
    def __init__(self, obs_dim: int, state_dim: int = 15, hidden_dim: int = 32):
        super(TrajectoryEncoder, self).__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        # x is expected to be [batch, seq_len, obs_dim]
        # output the initial latent state z0 to feed into ODE
        _, h_n = self.gru(x)
        z0 = self.fc(h_n[-1]) # Use last hidden state
        # Ensure z0 is strictly positive since it represents biological concentrations
        return torch.nn.functional.softplus(z0)


class ComplexityNeuralODE(nn.Module):
    """
    Full architecture: Encoder -> ODE Solver
    """
    def __init__(self, obs_dim: int = 15, state_dim: int = 15, hidden_dim: int = 64):
        super(ComplexityNeuralODE, self).__init__()
        self.encoder = TrajectoryEncoder(obs_dim=obs_dim, state_dim=state_dim)
        self.odef = ODEF(state_dim=state_dim, hidden_dim=hidden_dim)
        self.state_dim = state_dim

    def forward(self, series_data: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """
        Encode historical data and extrapolate future states.
        
        Args:
            series_data: History tensor [batch, seq_length, obs_dim]
            t_span: Evaluation times scalar array shape [num_timepoints]
            
        Returns:
            trajectory: Tensor of shape [batch, num_timepoints, state_dim]
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is required to execute the continuous forward pass.")

        # 1. Map sparse data to latent z0
        z0 = self.encoder(series_data)
        
        # 2. Integrate continuous vector field forward in time
        # odeint returns [timepoints, batch, state_dim]
        traj = odeint(self.odef, z0, t_span, method='rk4')
        
        # Rearrange to [batch, timepoints, state_dim]
        return traj.permute(1, 0, 2)


def compute_complexity_loss(predicted_traj: torch.Tensor, target_traj: torch.Tensor, l2_weight: float = 1.0, reg_weight: float = 0.1) -> torch.Tensor:
    """
    Custom loss function combining point-wise trajectory L2 (MSE) loss
    with an optional temporal entropy penalty (to penalize collapse).
    """
    # Standard Mean-Squared Error between trajectory points
    mse_loss = nn.MSELoss()(predicted_traj, target_traj)
    
    # ── Structural Stability Regularization (dZ/dt norm) ──
    # We penalize extreme high-frequency variations in the predicted trajectory
    # to encourage finding the simplest vector field that explains the data
    diffs = predicted_traj[:, 1:, :] - predicted_traj[:, :-1, :]
    smoothness_reg = torch.mean(torch.abs(diffs))
    
    return l2_weight * mse_loss + reg_weight * smoothness_reg

