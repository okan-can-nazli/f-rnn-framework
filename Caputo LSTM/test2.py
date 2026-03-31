# test_echo_caputo.py
import numpy as np
from CaputoLstm import LSTMCell




#np.random.seed(12345)



# --- CONFIGURATION ---
INPUT_DIM = 1
HIDDEN_DIM = 64
OUTPUT_DIM = 1
EPOCHS = 3000
LEARNING_RATE = 0.1
SIGMA = 1  # ← TEST: 1.0, 0.7, 0.5, 0.3

# --- DATA ---
x_sequence = np.array([[1], [0], [0], [0], [0]])
y_targets = np.array([[0], [0], [0], [0], [1]])

# --- TRAINING ---
lstm = LSTMCell(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
stm_init = np.zeros((HIDDEN_DIM, 1))
ltm_init = np.zeros((HIDDEN_DIM, 1))

for epoch in range(EPOCHS):
    stm_outputs, caches = lstm.forward_sequence(x_sequence, stm_init, ltm_init)
    
    dy_preds = []
    total_loss = 0
    
    for t in range(len(x_sequence)):
        pred = lstm.predict(stm_outputs[t])
        error = 2 * (pred - y_targets[t]) / len(x_sequence)
        dy_preds.append(error)
        total_loss += np.mean((pred - y_targets[t])**2)

    if epoch % 300 == 0:
        print(f"Epoch {epoch:04d} | Loss: {total_loss/len(x_sequence):.6f}")

    grads = lstm.backward_sequence(dy_preds, stm_outputs, caches, sigma=SIGMA)
    lstm.update_weights(grads, LEARNING_RATE)

print(f"\n=== CAPUTO LSTM (sigma={SIGMA}) ===")
final_outputs, _ = lstm.forward_sequence(x_sequence, stm_init, ltm_init)
print("Input\tTarget\tPrediction")
for i in range(len(x_sequence)):
    p = lstm.predict(final_outputs[i])
    print(f"{x_sequence[i][0]:.0f}\t{y_targets[i][0]:.0f}\t{p[0][0]:.4f}")