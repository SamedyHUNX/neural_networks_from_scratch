# Neural Network From Scratch

A pure NumPy implementation of a deep learning framework built from the ground up for educational purposes. This project demonstrates how neural networks work internally by implementing every component—layers, activations, loss functions, optimizers, and backpropagation—without relying on high-level frameworks like TensorFlow or PyTorch.

## Features

- **Pure NumPy Implementation**: Every operation is explicit and transparent
- **Complete Neural Network Framework**: Fully functional deep learning library
- **Multiple Layer Types**: Dense layers, Dropout regularization
- **Rich Activation Functions**: ReLU, Softmax, Sigmoid, Linear
- **Various Loss Functions**: Categorical/Binary Cross-Entropy, MSE, MAE
- **Modern Optimizers**: SGD with momentum, Adam, RMSprop, Adagrad
- **Regularization Support**: L1/L2 weight regularization, Dropout
- **Model Persistence**: Save and load trained models
- **Educational Notebooks**: Step-by-step chapters building from basics to advanced concepts

## Project Structure

```
neural_network_fs/
├── modules/                      # Core framework implementation
│   ├── model.py                 # Main Model class
│   ├── layers.py                # Layer implementations
│   ├── activation_functions.py  # Activation functions
│   ├── losses.py                # Loss functions
│   ├── optimizers.py            # Optimization algorithms
│   └── accuracy.py              # Accuracy metrics
├── chapter02_coding_our_first_neurons/
├── chapter03_adding_layers/
├── chapter04_activation_functions/
├── ...
├── chapter18_model_object/
├── chapter19_data_preparation/  # Fashion MNIST training
├── WARP.md                      # Architecture documentation
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd neural_network_fs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, install manually:
```bash
pip install numpy opencv-python jupyter matplotlib
```

## Quick Start

### Using the Framework

```python
import sys, os
sys.path.append(os.path.abspath('modules'))

from model import Model
from layers import Layer_Dense, Layer_Dropout
from activation_functions import Activation_ReLU, Activation_Softmax
from losses import Loss_CategoricalCrossentropy
from optimizers import Optimizer_Adam
from accuracy import Accuracy_Categorical

# Create model
model = Model()

# Add layers
model.add(Layer_Dense(784, 128))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.2))
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Configure model
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-4),
    accuracy=Accuracy_Categorical()
)

# Finalize model
model.finalize()

# Train
model.train(X_train, y_train, epochs=10, batch_size=128,
            validation_data=(X_val, y_val))

# Make predictions
predictions = model.predict(X_test, batch_size=128)

# Save model
model.save('my_model.model')

# Load model
loaded_model = Model.load('my_model.model')
```

## Learning Path

The project is organized into chapters that progressively build understanding:

1. **Chapters 2-3**: Basic neurons and layers
2. **Chapter 4**: Activation functions (ReLU, Softmax)
3. **Chapters 5-6**: Loss functions and optimization
4. **Chapters 7-9**: Calculus fundamentals and backpropagation
5. **Chapter 10**: Advanced optimizers (Adam, RMSprop)
6. **Chapter 11**: Training/validation splits
7. **Chapter 14**: Regularization techniques
8. **Chapters 16-17**: Binary classification and regression
9. **Chapter 18**: Complete Model class
10. **Chapter 19**: Real-world dataset (Fashion MNIST)

### Running Jupyter Notebooks

Navigate to any chapter directory and launch Jupyter:

```bash
cd chapter19_data_preparation
jupyter notebook code.ipynb
```

Run cells sequentially to see concepts in action.

## Fashion MNIST Example

The final chapter includes a complete Fashion MNIST classifier:

```bash
cd chapter19_data_preparation
jupyter notebook code.ipynb
```

The dataset will be automatically downloaded on first run (~30MB).

## Key Concepts Implemented

### Linked Layer Architecture
Each layer maintains `prev` and `next` references, enabling automatic gradient flow:
```python
# Forward pass
for layer in self.layers:
    layer.forward(layer.prev.output, training)

# Backward pass
for layer in reversed(self.layers):
    layer.backward(layer.next.dinputs)
```

### Optimized Softmax-CrossEntropy
Combined activation and loss for numerical stability:
```python
# Analytical gradient: predictions - y_true
self.dinputs = dvalues.copy()
self.dinputs[range(samples), y_true] -= 1
self.dinputs /= samples
```

### Batch Processing
All operations are vectorized for efficiency:
```python
output = inputs @ weights + biases  # Handles batches automatically
```

## Model Persistence

### Save/Load Parameters Only
```python
model.save_parameters('weights.params')
model.load_parameters('weights.params')
```

### Save/Load Complete Model
```python
model.save('complete_model.model')
loaded_model = Model.load('complete_model.model')
```

## Architecture Details

For comprehensive architecture documentation, see [WARP.md](WARP.md), which includes:
- Detailed component descriptions
- Data flow diagrams
- Implementation patterns
- Design decisions and trade-offs

## Dependencies

- **NumPy**: Core numerical operations
- **OpenCV (cv2)**: Image loading for Fashion MNIST
- **Jupyter**: Interactive notebooks
- **Matplotlib** (optional): Visualization

## Limitations

This is an **educational implementation** with intentional limitations:

- **CPU-only**: No GPU acceleration
- **Performance**: Not optimized for production use
- **Layer types**: Limited to dense and dropout layers
- **No automatic differentiation**: All gradients manually derived

For production use, consider TensorFlow, PyTorch, or JAX.

## Contributing

This is an educational project. If you find bugs or have suggestions for improving clarity, feel free to open an issue or submit a pull request.

## License

This project is open source and available for educational purposes.

## Acknowledgments

Built following neural network fundamentals, implementing every component from scratch to provide a clear understanding of how deep learning frameworks work internally.

## Resources

- [WARP.md](WARP.md) - Detailed architecture documentation
- Individual chapter notebooks - Step-by-step explanations
- Module source code - Commented implementations

## Contact

For questions or feedback about this educational project, please open an issue in the repository.
