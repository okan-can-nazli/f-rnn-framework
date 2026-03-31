# test_caputo_rnn.py
import numpy as np
from CaputoRNN import RNNCell

print("--- TESTING CAPUTO RNN ---")

# Dataset
seq_length = 10
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps).reshape(-1, 1)

x_seq = data
targets = np.roll(data, -1)
targets[-1] = 0.0

# Model
SIGMA = 1 
np.random.seed(42)
rnn = RNNCell(input_size=1, hidden_size=5, output_size=1, activation='relu', lr=0.01, sigma=SIGMA)
h_init = np.zeros((1, 5))

print(f"Sigma: {SIGMA}")
print(f"Initial W1[0,0]: {rnn.W1[0,0]:.6f}")

# Train
for epoch in range(1, 101):
    outputs, caches = rnn.forward_sequence(x_seq, h_init)  # ← returns tuple
    
    loss = np.mean([(pred - targ)**2 for pred, targ in zip(outputs, targets)])
    d_outputs = [(pred - targ) for pred, targ in zip(outputs, targets)]
    
    rnn.backward_sequence(d_outputs, caches)  # ← add sigma
    
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | Loss: {loss:.6f}")

print(f"\nFinal W1[0,0]: {rnn.W1[0,0]:.6f}")
