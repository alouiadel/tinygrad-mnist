# Standard library imports for time handling and file operations
from datetime import datetime
from pathlib import Path
import json

# TinyGrad imports for neural network operations and model management
from tinygrad import Tensor, nn, Device
from tinygrad.nn.state import safe_save, safe_load

# Training hyperparameters and configuration
# How many images to process in one training step
BATCH_SIZE = 128
# Total number of training steps to perform
NUM_STEPS = 7000
# How often to evaluate model performance
EVAL_INTERVAL = 100
# Directory to store trained models
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)


class TrainingStats:
    """Tracks and stores training metrics like accuracy, loss, and timing."""

    def __init__(self):
        # Keep track of the best accuracy seen during training
        self.best_acc = 0.0
        # Record when training started for timing
        self.start_time = datetime.now()
        # Store training history for analysis
        self.history = []

    def update(self, step, loss, acc):
        """Records training metrics and updates best accuracy if necessary."""
        # Add current metrics to training history
        self.history.append({"step": step, "loss": loss, "acc": acc})
        # Update best accuracy if current accuracy is better
        self.best_acc = max(self.best_acc, acc)

    def get_elapsed_time(self):
        """Returns formatted training duration."""
        # Calculate time elapsed since training started
        return datetime.now() - self.start_time


def create_training_step(model, X_train, Y_train, optim):
    """Creates a training step function that performs one SGD update."""

    def step():
        # Enable training mode (activates dropout, etc.)
        Tensor.training = True
        # Randomly select batch_size samples for training
        samples = Tensor.randint(BATCH_SIZE, high=X_train.shape[0])
        X, Y = X_train[samples], Y_train[samples]
        # Clear previous gradients
        optim.zero_grad()
        # Forward pass -> compute loss -> backward pass -> update weights
        loss = model(X).sparse_categorical_crossentropy(Y).backward()
        optim.step()
        return loss

    return step


def evaluate_model(model, X_test, Y_test):
    """Computes model accuracy on the test set."""
    # Disable training mode for evaluation (no dropout)
    Tensor.training = False
    # Compare predicted digits with true labels and compute accuracy
    return (model(X_test).argmax(axis=1) == Y_test).mean().item()


def save_model(model, stats, filename=None):
    """Save model weights and training info using TinyGrad's native format."""
    # Generate filename with timestamp and accuracy if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        acc = f"{stats.best_acc * 100:.2f}".replace(".", "p")
        filename = f"mnist_model_{timestamp}_acc{acc}"

    # Create directory for model files
    save_path = MODELS_DIR / filename
    save_path.mkdir(exist_ok=True)

    # Save model weights in safetensors format
    weights_path = save_path / "weights.safetensors"
    params = nn.state.get_parameters(model)
    param_dict = {f"param_{i}": param for i, param in enumerate(params)}
    safe_save(param_dict, weights_path)

    # Save training statistics and model info
    training_info = {
        "best_accuracy": stats.best_acc,
        "training_time": str(stats.get_elapsed_time()),
        "total_steps": len(stats.history),
        "batch_size": BATCH_SIZE,
        "device": str(Device.DEFAULT),
        "history": stats.history,
        "num_parameters": len(params),
    }

    # Write training info to JSON file
    with open(save_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    return save_path


def load_model(model_path, model_class):
    """Load a saved model using TinyGrad's native format."""
    # Create a new instance of the model
    model = model_class()

    # Load saved weights from safetensors file
    weights_path = Path(model_path) / "weights.safetensors"
    param_dict = safe_load(weights_path)

    # Load training info from JSON file
    with open(Path(model_path) / "training_info.json", "r") as f:
        training_info = json.load(f)
    num_params = training_info["num_parameters"]

    # Reconstruct model parameters in correct order
    params = [param_dict[f"param_{i}"] for i in range(num_params)]
    # Get current model parameters
    model_params = nn.state.get_parameters(model)
    # Update model with loaded parameters
    for model_param, loaded_param in zip(model_params, params):
        # Move loaded parameter to the same device as the model parameter
        loaded_param = loaded_param.to(model_param.device)
        model_param.assign(loaded_param)

    return model, training_info
