from tinygrad.nn.datasets import mnist
from colorama import Fore, Style


def load_data():
    """Downloads and prepares MNIST dataset for training."""
    print(f"{Fore.GREEN}Loading MNIST dataset...{Style.RESET_ALL}")
    X_train, Y_train, X_test, Y_test = mnist()
    print(
        f"{Fore.GREEN}✓ Loaded dataset with shapes: {X_train.shape}, {Y_train.shape}{Style.RESET_ALL}"
    )
    return X_train, Y_train, X_test, Y_test
