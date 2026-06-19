# TinyGrad provides the building blocks for neural networks
# Tensor: Multi-dimensional array for storing and processing data
# nn: Neural network modules like layers and operations
from tinygrad import Tensor, nn


# This neural network is based on LeNet-5, a classic CNN architecture
# CNN = Convolutional Neural Network, great for image recognition tasks
# MNIST = Dataset of handwritten digits (0-9) in 28x28 pixel grayscale images
class Model:
    """LeNet-5 inspired CNN architecture for MNIST digit classification."""

    def __init__(self):
        # First convolutional layer:
        # - Input: 1 channel (grayscale images) of size 28x28
        # - Output: 32 feature maps (learns 32 different patterns)
        # - Kernel: 3x3 sliding window that scans the image
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3, 3))

        # Second convolutional layer:
        # - Input: 32 channels from previous layer
        # - Output: 64 feature maps (learns more complex patterns)
        # - Kernel: 3x3 sliding window that combines previous features
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))

        # Final fully connected (linear) layer:
        # - Input: 1600 flattened features (result of previous convolutions and pooling)
        # - Output: 10 neurons (one for each digit 0-9)
        # - Each output neuron's value represents the confidence for that digit
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass: processes the image through the network to make a prediction."""
        # First processing block:
        # 1. Apply convolution to find basic patterns (edges, curves)
        # 2. ReLU activation: keeps positive values, sets negatives to zero
        # 3. Max pooling: reduces size by keeping strongest features (2x2 window)
        x = self.l1(x).relu().max_pool2d((2, 2))

        # Second processing block:
        # 1. Apply convolution to find more complex patterns (shapes, parts of digits)
        # 2. ReLU activation: keeps positive values, sets negatives to zero
        # 3. Max pooling: further reduces size by keeping strongest features
        x = self.l2(x).relu().max_pool2d((2, 2))

        # Final classification:
        # 1. Flatten: convert 2D feature maps into 1D vector
        # 2. Dropout: randomly disable 50% of neurons to prevent overfitting
        # 3. Linear layer: compute final scores for each digit (0-9)
        return self.l3(x.flatten(1).dropout(0.5))
