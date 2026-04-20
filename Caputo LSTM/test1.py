from CaputoLstm import LSTMCell
import numpy as np



np.random.seed(12345)




x_sequence = np.array([[0], [1], [0], [1], [1]])
stm_size = 128
sigma=1
# Test with sigma=1.0 (should be same as standard)
lstm = LSTMCell(input_size=1, stm_size=stm_size, output_size=1)

for step in range(1000):
    stm_outputs, caches = lstm.forward_sequence(
        x_sequence,
        stm_init=np.zeros((stm_size, 1)),
        ltm_init=np.zeros((stm_size, 1))
    )
    
    predictions = np.array([lstm.predict(stm) for stm in stm_outputs]).reshape(-1, 1)
    loss = np.mean((predictions - x_sequence)**2)
    
    if step % 200 == 0:
        print(f"Step {step}, Loss: {loss:.6f}")
    
    dpredictions = 2 * (predictions - x_sequence) / len(x_sequence)
    dy_preds = [dp.reshape(1, 1) for dp in dpredictions]
    
    grads = lstm.backward_sequence(dy_preds, stm_outputs, caches, sigma)
    lstm.update_weights(grads, learning_rate=0.1)

print("\nFinal Output:")
print(f"for sigma:{sigma}")
for inp, pred in zip(x_sequence, predictions):
    print(f"{inp[0]:.1f} -> {pred[0]:.4f}")