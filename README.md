# Fractional-Order RNN Framework

## Overview
Extension of standard recurrent neural networks (RNN, LSTM, GRU) using Caputo fractional derivatives to investigate fractional-order memory dynamics in sequence modeling.

## Implementation
- **Caputo Derivative**: Implements fractional calculus (α ∈ (0,1)) in backward pass
- **Three Architectures**: Fractional versions of RNN, LSTM, and GRU
- **Pure NumPy**: No auto-differentiation frameworks, all gradients manually derived

## Technical Details
The Caputo fractional derivative is applied to activation function gradients during backpropagation, introducing a power-law memory effect across the temporal sequence.

**Verification**: When σ = 1.0, the implementation reduces to standard gradient computation, validating correctness.

## Repository Structure
Caputo RNN/     - Fractional RNN implementation
Caputo LSTM/    - Fractional LSTM implementation
Caputo GRU/     - Fractional GRU implementation

## Related Work
Standard implementations: [lstm-from-scratch](https://github.com/okan-can-nazli/lstm-from-scratch)

## Tech Stack
- Python 3
- NumPy
- Math (gamma function)
