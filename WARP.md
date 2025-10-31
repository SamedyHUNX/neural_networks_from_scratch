# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an educational neural network implementation built from scratch using NumPy. The project demonstrates building a complete deep learning framework without relying on high-level libraries like TensorFlow or PyTorch. It progresses through chapters that incrementally build up neural network concepts from basic neurons to a full model implementation trained on the Fashion MNIST dataset.

## Project Structure

The repository is organized into:

- **`modules/`** - Core neural network framework implementation (production code)
- **`chapterXX_*/`** - Educational Jupyter notebooks demonstrating concepts step-by-step
- **`fashion_mnist_images/`** - Training dataset (gitignored, downloaded on demand)

## Core Architecture

The framework follows an object-oriented design with these primary components:

### 1. Model (`modules/model.py`)

The `Model` class is the orchestrator that:
- Manages the network architecture as a linked list of layers
- Chains forward and backward propagation through all layers
- Handles training loops with mini-batching support
- Implements model persistence (save/load parameters or entire model)
- Optimizes softmax+categorical cross-entropy with a combined activation/loss class

**Key Design Pattern**: Each layer maintains `prev` and `next` references, enabling automatic gradient flow during backpropagation without explicit layer management.

### 2. Layers (`modules/layers.py`)

- **`Layer_Input`**: Pass-through layer that provides input interface
- **`Layer_Dense`**: Fully connected layer with:
  - Weight initialization: `weights = 0.1 * np.random.randn(n_inputs, n_neurons)`
  - Forward: `output = inputs @ weights + biases`
  - Backward: Computes gradients for weights, biases, and inputs
  - L1/L2 regularization support on both weights and biases
- **`Layer_Dropout`**: Regularization layer using inverted dropout
  - Only active during training (`training=True`)
  - Uses binary mask scaled by `1 / (1 - dropout_rate)`

### 3. Activation Functions (`modules/activation_functions.py`)

All activations implement `forward()`, `backward()`, and `predictions()` methods:

- **ReLU**: `max(0, x)` - Standard for hidden layers
- **Softmax**: Normalized exponentials for multi-class classification
- **Sigmoid**: Binary classification
- **Linear**: Regression tasks (identity function)

**Implementation Note**: Softmax backward pass uses full Jacobian matrix calculation per sample.

### 4. Loss Functions (`modules/losses.py`)

Base `Loss` class provides:
- Accumulated loss tracking across batches
- Regularization loss calculation from all trainable layers
- Sample-wise loss computation

Specific losses:
- **`Loss_CategoricalCrossentropy`**: Multi-class classification
- **`Loss_BinaryCrossentropy`**: Binary classification  
- **`Loss_MeanSquaredError`**: Regression (L2)
- **`Loss_MeanAbsoluteError`**: Regression (L1)
- **`Activation_Softmax_Loss_CategoricalCrossentropy`**: Combined activation/loss for efficient gradient computation

### 5. Optimizers (`modules/optimizers.py`)

All optimizers follow the same interface:
1. `pre_update_params()` - Apply learning rate decay
2. `update_params(layer)` - Update layer weights/biases
3. `post_update_params()` - Increment iteration counter

Available optimizers:
- **SGD**: Vanilla stochastic gradient descent with optional momentum
- **Adagrad**: Adaptive learning rates with per-parameter cache
- **RMSprop**: Moving average of squared gradients (rho=0.9)
- **Adam**: Combines momentum and RMSprop with bias correction

### 6. Accuracy Metrics (`modules/accuracy.py`)

- **`Accuracy_Categorical`**: For classification (compares argmax)
- **`Accuracy_Regression`**: Precision-based accuracy (within std/250 of target)

Both support accumulated accuracy tracking across batches.

## Data Flow

### Training Pipeline

```
1. Model.train(X, y, epochs, batch_size)
   ├─ Initialize accuracy metric
   ├─ For each epoch:
   │   ├─ For each mini-batch:
   │   │   ├─ Forward pass: input → layers → output
   │   │   ├─ Calculate loss (data + regularization)
   │   │   ├─ Backward pass: loss → layers → gradients
   │   │   └─ Optimizer updates all trainable layers
   │   └─ Print epoch statistics
   └─ Optional validation evaluation
```

### Forward Pass Chain

```
Input → Layer_Input.forward()
     ↓
     Layer_Dense.forward() → Activation.forward()
     ↓ (repeated for each layer)
     Layer_Dense.forward() → Activation_Softmax.forward()
     ↓
     Output (predictions)
```

Each layer reads `layer.prev.output` and writes to `layer.output`.

### Backward Pass Chain

```
Loss.backward(output, y_true)  [or combined softmax-loss]
     ↓
     Activation.backward(layer.next.dinputs)
     ↓
     Layer_Dense.backward(layer.next.dinputs)
     ↓ (repeated in reverse)
     Computes: dweights, dbiases, dinputs for each layer
```

Gradients flow through `dinputs` from each layer to its predecessor.

## Chapter Structure

Chapters are educational notebooks building concepts incrementally:

- **Chapter 2-3**: Basic neuron implementation, layer stacking
- **Chapter 4**: Activation functions (ReLU, Softmax)
- **Chapter 5-6**: Loss functions, optimization concepts
- **Chapter 7-9**: Derivatives, backpropagation mathematics
- **Chapter 10**: Optimizer implementations (SGD, Adam, etc.)
- **Chapter 11**: Train/validation split, overfitting detection
- **Chapter 14**: Regularization (L1/L2, Dropout)
- **Chapter 16-17**: Binary classification and regression
- **Chapter 18**: Complete Model class implementation
- **Chapter 19**: Real dataset (Fashion MNIST) training

Each chapter has a `code.ipynb` notebook that imports from `../modules/`.

## Working with the Codebase

### Import Pattern in Notebooks

All notebooks use this pattern to import the framework:

```python
import sys, os
sys.path.append(os.path.abspath('../modules'))

from model import Model
from layers import Layer_Dense, Layer_Dropout
from activation_functions import Activation_ReLU, Activation_Softmax
from losses import Loss_CategoricalCrossentropy
from optimizers import Optimizer_Adam
from accuracy import Accuracy_Categorical
```

### Typical Model Construction

```python
# Instantiate model
model = Model()

# Add layers
model.add(Layer_Dense(input_dim, 128))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.2))
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Configure
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-4),
    accuracy=Accuracy_Categorical()
)

# Finalize (links layers, creates combined softmax-loss if applicable)
model.finalize()

# Train
model.train(X_train, y_train, epochs=10, batch_size=128,
            validation_data=(X_val, y_val))

# Predict
predictions = model.predict(X_test, batch_size=128)
```

### Model Persistence

```python
# Save parameters only (smaller, requires model reconstruction)
model.save_parameters('fashion_mnist.params')
model.load_parameters('fashion_mnist.params')

# Save entire model (includes architecture, optimizer state)
model.save('fashion_mnist.model')
loaded_model = Model.load('fashion_mnist.model')
```

## Key Implementation Details

### Regularization Integration

Regularization is specified per-layer but calculated globally:
1. Layers store regularization hyperparameters (`weight_regularizer_l1`, etc.)
2. During backward pass, layers compute regularization gradients and add to `dweights`/`dbiases`
3. Loss class computes total regularization loss by iterating all trainable layers
4. Total loss = data_loss + regularization_loss

### Softmax-CrossEntropy Optimization

The combined `Activation_Softmax_Loss_CategoricalCrossentropy` class implements the analytical gradient:
```python
# Instead of: softmax'(x) * cross_entropy'(softmax(x))
# Uses simplified: predictions - y_true
self.dinputs = dvalues.copy()
self.dinputs[range(samples), y_true] -= 1
self.dinputs /= samples
```

This avoids numerical instability and is significantly faster.

### Batch Processing

All operations are vectorized across the batch dimension (axis 0):
- Forward: `output = inputs @ weights + biases` handles batch automatically
- Backward: Gradient accumulation uses `np.sum(..., axis=0, keepdims=True)` for biases
- Loss: Computed per-sample, then averaged

### Training vs Inference Mode

Layers receive a `training` boolean flag:
- **Dropout**: Binary mask applied only when `training=True`
- **Future extensibility**: Could support BatchNorm, data augmentation, etc.

## Dependencies

- **NumPy**: Core numerical operations (matrix multiplication, activation functions)
- **OpenCV** (`cv2`): Image loading for Fashion MNIST dataset
- **Pickle**: Model serialization
- **Jupyter**: For educational notebooks

## Running Experiments

Since this is a learning-focused repository without standard test scripts:

1. **Navigate to a chapter directory**
2. **Open the Jupyter notebook**: `jupyter notebook code.ipynb`
3. **Run cells sequentially** - each chapter builds on previous concepts
4. **Experiment by modifying**:
   - Layer dimensions and depths
   - Learning rates and optimizers
   - Regularization strengths
   - Activation functions

The framework is designed for experimentation and understanding, not production deployment.

## Strengths of This Design

- **Pure NumPy**: No magic - every operation is explicit and educational
- **Modular**: Each component (layer, activation, loss) is independently understandable
- **Linked architecture**: Automatic gradient flow without manual layer tracking
- **Extensible**: Adding new layers/activations/optimizers follows clear patterns
- **Accumulation support**: Efficient batch training with proper metric averaging

## Limitations

- **Performance**: Not optimized for speed (pure Python loops in some backward passes)
- **No GPU support**: NumPy only runs on CPU
- **Limited layer types**: No convolutions, recurrence, attention, etc.
- **No automatic differentiation**: All gradients manually derived
- **Single-threaded**: No data loading parallelism

This is an educational implementation prioritizing clarity over performance.
