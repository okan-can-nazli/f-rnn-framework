import numpy as np
from CaputoGRU import GRUCell 

# np.random.seed(12345)



def run_fractional_test():
    X_train = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]])
    Y_true = np.array([[0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]])

    input_size = 1
    hidden_size = 16
    output_size = 1
    gru = GRUCell(input_size, hidden_size, output_size)

    epochs = 2000
    learning_rate = 0.05
    sigma = 1  # Fractional order parameter

    print(f"{'Epoch':<10} | {'Loss':<10}")
    print("-" * 25)

    for epoch in range(epochs):
        Y_pred = gru.forward_sequence(X_train)
        
        loss = np.mean((Y_pred - Y_true) ** 2)
        
        d_outputs = 2 * (Y_pred - Y_true) / len(Y_true)
        gru.backward(d_outputs, lr=learning_rate, sigma=sigma)
        
        if epoch % 200 == 0:
            print(f"{epoch:<10} | {loss:.8f}")

    print("\n" + "="*40)
    print(f"PREDICTIONS (SIGMA = {sigma})")
    print("="*40)
    
    final_preds = gru.forward_sequence(X_train)
    for i in range(len(X_train)):
        print(f"input: {X_train[i][0]:.1f} | target: {Y_true[i][0]:.1f} | prediction: {final_preds[i][0]:.4f}")

if __name__ == "__main__":
    run_fractional_test()