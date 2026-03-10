# -*- coding: utf-8 -*-
"""
Neural-ODE Training & Digital Twin Memory Integration Pipeline
==============================================================

This script demonstrates the end-to-end continuous learning flow for Project Confluence:
1. Generate synthetic clinical time-series observations representing tumor evolution.
2. Train a Neural-ODE model to learn the topological vector field governing the states.
3. Pipe the continuous output trajectory directly into the DigitalTwinMemory module 
   (Graphiti/Cognee adapters) to extract 19D Complexity metrics.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Bind project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from models.neural_ode import ComplexityNeuralODE, compute_complexity_loss, TORCHDIFFEQ_AVAILABLE
    from agents.digital_twin_memory import DigitalTwinMemory
    from models.ode_system import ComplexAttractorODE, ExtendedParams
except ImportError as e:
    print(f"❌ Initialization Error: {e}")
    sys.exit(1)

# Training Hyperparameters
EPOCHS = 50
BATCH_SIZE = 4
SEQ_LEN = 20
STATE_DIM = 15
LEARNING_RATE = 1e-3


def generate_synthetic_patient_data(num_samples: int = BATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic clinical histories using the TNBC ComplexAttractorODE baseline.
    Returns:
       batch_history (Tensor): [BATCH, SEQ_LEN, STATE_DIM] Sparse historical observations.
       batch_future  (Tensor): [BATCH, SEQ_LEN, STATE_DIM] Future target trajectory to reconstruct.
    """
    print("🧬 Generating synthetic baseline trajectories...")
    # Initialize the rigid biological model
    params = ExtendedParams()
    biological_system = ComplexAttractorODE(params=params)
    
    # We will compute t_span for evaluation
    dt = 1.0 # Days per step
    t_eval = np.arange(0, SEQ_LEN * 2 * dt, dt) 
    
    batch_history = []
    batch_future = []
    
    for i in range(num_samples):
        # Slightly perturb initial state to simulate patient variance
        z0 = biological_system.healthy_initial_state()
        z0 += np.random.normal(0, 0.1, size=len(z0))
        z0 = np.clip(z0, 0.01, None) # Biological states are positive
        
        # Integrate rigid PDE
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(biological_system.rhs, (0, t_eval[-1]), z0, t_eval=t_eval, method='LSODA')
            trajectory = sol.y.T # Shape: [SEQ_LEN * 2, STATE_DIM]
            
            # Add clinical measurement noise
            noisy_traj = trajectory + np.random.normal(0, 0.05, size=trajectory.shape)
            noisy_traj = np.maximum(noisy_traj, 0.0) # Ensure positivity
            
            # Split historical vs future target
            batch_history.append(noisy_traj[:SEQ_LEN])
            batch_future.append(noisy_traj[SEQ_LEN:])
            
        except Exception as e:
            print(f"Warning: ODE integration failed during generation: {e}")
            # Fallback random initialization if scipy fails
            batch_history.append(np.random.rand(SEQ_LEN, STATE_DIM))
            batch_future.append(np.random.rand(SEQ_LEN, STATE_DIM))
            
    history_tensor = torch.tensor(np.array(batch_history), dtype=torch.float32)
    future_tensor = torch.tensor(np.array(batch_future), dtype=torch.float32)
    
    # time steps relative to prediction start (T=0)
    t_span = torch.arange(0, SEQ_LEN, dtype=torch.float32) 
    
    return history_tensor, future_tensor, t_span


def run_training_pipeline():
    """Main execution block."""
    if not TORCHDIFFEQ_AVAILABLE:
        print("⚠️ `torchdiffeq` is not installed! Cannot execute Neural-ODE continuous integration.")
        print("Run: pip install torchdiffeq")
        return

    print(f"🚀 Initializing Neural-ODE Architecture (Epochs={EPOCHS})")
    
    # 1. Dataset Initialization
    history_tensor, future_tensor, t_span = generate_synthetic_patient_data()
    
    # 2. Model Initialization
    model = ComplexityNeuralODE(obs_dim=STATE_DIM, state_dim=STATE_DIM, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Training Loop ---")
    
    # 3. Training Loop Iterator
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Unroll the continuous trajectory
        predicted_future = model(history_tensor, t_span)
        
        # Calculate divergence
        loss = compute_complexity_loss(predicted_future, future_tensor, l2_weight=1.0, reg_weight=0.05)
        
        # Backpropagate through time
        loss.backward()
        
        # Stabilize Gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}/{EPOCHS} | Complexity Loss: {loss.item():.4f}")

    print("\n✅ Neural-ODE Training Finalized.")
    print("\n--- Digital Twin Memory Evaluation ---")
    
    # 4. Integrate with Memory Adapters
    print("🧠 Extracting topological relationships from learned trajectory...")
    patient_id = "PT-NODE-001"
    memory_agent = DigitalTwinMemory(patient_id=patient_id)
    
    # Generate an evaluation state
    with torch.no_grad():
        test_history, _, eval_t_span = generate_synthetic_patient_data(num_samples=1)
        # Pull single generated batch trajectory
        eval_trajectory = model(test_history, eval_t_span).squeeze(0).numpy() # Shape [SEQ_LEN, STATE_DIM]
        time_arr = eval_t_span.numpy()
        
    res = memory_agent.process_neural_trajectory(eval_trajectory, time_arr)
    print("\n[Adapter Telemetry]")
    print(f"• Graphiti Temporal Shifts Logged: {res['temporal_events_recorded']}")
    print(f"• Cognee Structural Edges Mapped: {res['unique_correlations_mapped']}")
    
    print("\n[Final 19D Complexity Vector Export]")
    features = memory_agent.get_memory_features()
    for feat, val in features.items():
        print(f" - {feat:25s} : {val:.4f}")
        

if __name__ == "__main__":
    run_training_pipeline()
