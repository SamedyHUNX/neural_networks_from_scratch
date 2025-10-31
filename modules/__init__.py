"""
Neural Network From Scratch
A pure NumPy implementation of a deep learning framework for educational purposes.
"""

from .model import Model
from .layers import Layer_Dense, Layer_Dropout, Layer_Input
from .activation_functions import (
    Activation_ReLU,
    Activation_Softmax,
    Activation_Sigmoid,
    Activation_Linear
)
from .losses import (
    Loss,
    Loss_CategoricalCrossentropy,
    Loss_BinaryCrossentropy,
    Loss_MeanSquaredError,
    Loss_MeanAbsoluteError,
    Activation_Softmax_Loss_CategoricalCrossentropy
)
from .optimizers import (
    Optimizer_SGD,
    Optimizer_Adam,
    Optimizer_RMSprop,
    Optimizer_Adagrad
)
from .accuracy import (
    Accuracy,
    Accuracy_Categorical,
    Accuracy_Regression
)

__version__ = "0.1.0"

__all__ = [
    # Model
    "Model",
    
    # Layers
    "Layer_Dense",
    "Layer_Dropout",
    "Layer_Input",
    
    # Activations
    "Activation_ReLU",
    "Activation_Softmax",
    "Activation_Sigmoid",
    "Activation_Linear",
    
    # Losses
    "Loss",
    "Loss_CategoricalCrossentropy",
    "Loss_BinaryCrossentropy",
    "Loss_MeanSquaredError",
    "Loss_MeanAbsoluteError",
    "Activation_Softmax_Loss_CategoricalCrossentropy",
    
    # Optimizers
    "Optimizer_SGD",
    "Optimizer_Adam",
    "Optimizer_RMSprop",
    "Optimizer_Adagrad",
    
    # Accuracy
    "Accuracy",
    "Accuracy_Categorical",
    "Accuracy_Regression",
]
