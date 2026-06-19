#!/usr/bin/env python3
"""Simple MNIST training script using TinyGrad with a clean TUI."""

# Standard library imports for system, timing, and file operations
import sys
import timeit
import json
import os

# TinyGrad imports for neural network operations and JIT compilation
from tinygrad import nn, TinyJit
from colorama import Fore, Style

# Local module imports for model architecture and training components
from src.model import Model
from src.training import (
    TrainingStats,
    create_training_step,
    evaluate_model,
    save_model,
    load_model,
    BATCH_SIZE,
    NUM_STEPS,
    EVAL_INTERVAL,
    MODELS_DIR,
)

# Display utilities for terminal UI management
from src.display import (
    clear_screen,
    print_header,
    print_progress_bar,
    print_stats,
    show_cursor,
    hide_cursor,
    move_cursor,
)

# Data loading and prediction utilities
from src.data import load_data
from src.predict import predict_local_images


# Main training loop with model management, training, and evaluation functionality
def main():
    """Main training loop with progress tracking and live statistics display."""
    clear_screen()
    print_header(BATCH_SIZE, NUM_STEPS)

    # Check for and handle existing model loading
    saved_models = list(MODELS_DIR.glob("*"))
    if saved_models:
        print(
            f"\n{Fore.YELLOW}Found {len(saved_models)} saved model(s):{Style.RESET_ALL}"
        )
        for i, model_path in enumerate(saved_models):
            try:
                with open(os.path.join(model_path, "training_info.json"), "r") as f:
                    info = json.load(f)
                    acc = info.get("best_accuracy", "N/A")
                print(f"{i + 1}. {model_path.name} (Accuracy: {acc:.2%})")
            except (IOError, json.JSONDecodeError):
                print(f"{i + 1}. {model_path.name}")

        print(f"\n{Fore.CYAN}Options:{Style.RESET_ALL}")
        print("1. Load a saved model")
        print("2. Train a new model")
        choice = input(
            f"\n{Fore.CYAN}Enter your choice [1/2]:{Style.RESET_ALL} "
        ).strip()

        if choice == "1":
            model_idx = (
                int(input(f"{Fore.CYAN}Enter model number to load:{Style.RESET_ALL} "))
                - 1
            )
            if 0 <= model_idx < len(saved_models):
                model, info = load_model(saved_models[model_idx], Model)
                print(f"{Fore.GREEN}Model loaded successfully!{Style.RESET_ALL}")
                print(f"Best accuracy: {info['best_accuracy']:.2%}")
                predict_local_images(model)
                return
            else:
                print(f"{Fore.RED}Invalid model number.{Style.RESET_ALL}")
                return

    # Initialize training data and model components
    X_train, Y_train, X_test, Y_test = load_data()

    print(f"{Fore.GREEN}Initializing model...{Style.RESET_ALL}")
    model = Model()
    optim = nn.optim.Adam(nn.state.get_parameters(model))
    step_fn = create_training_step(model, X_train, Y_train, optim)

    # Evaluate initial model performance
    print(f"{Fore.YELLOW}Measuring initial performance...{Style.RESET_ALL}")
    initial_acc = evaluate_model(model, X_test, Y_test)
    print(f"Initial accuracy: {initial_acc * 100:.2f}%")

    # JIT compile the training step for better performance
    print(f"{Fore.GREEN}JIT compiling training step...{Style.RESET_ALL}")
    jit_step = TinyJit(step_fn)
    timings = timeit.repeat(jit_step, repeat=5, number=1)
    print(f"JIT compilation complete. Best timing: {min(timings) * 1000:.1f}ms")

    # Display training configuration and get user confirmation
    print(f"\n{Fore.YELLOW}Ready to start training:{Style.RESET_ALL}")
    print(f"• Will train for {NUM_STEPS:,} steps")
    print(f"• Batch size: {BATCH_SIZE}")
    print(f"• Evaluation every {EVAL_INTERVAL} steps")
    response = (
        input(f"\n{Fore.CYAN}Start training? [Y/n]:{Style.RESET_ALL} ").lower().strip()
    )
    if response not in ["", "y", "yes"]:
        print(f"{Fore.RED}Training cancelled.{Style.RESET_ALL}")
        return

    # Main training loop with progress tracking
    stats = TrainingStats()
    clear_screen()
    print_header(BATCH_SIZE, NUM_STEPS)

    try:
        for step in range(NUM_STEPS):
            loss = jit_step()

            # Update visual progress indicators
            move_cursor(0, 10)
            print(print_progress_bar(step + 1, NUM_STEPS))

            if step % EVAL_INTERVAL == 0:
                acc = evaluate_model(model, X_test, Y_test)
                stats.update(step, loss.item(), acc)
                print_stats(
                    step,
                    loss.item(),
                    acc,
                    stats.best_acc,
                    stats.get_elapsed_time(),
                    NUM_STEPS,
                )

            hide_cursor()
            sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Training interrupted by user.{Style.RESET_ALL}")
    finally:
        show_cursor()

    # Display final training results
    print(f"\n{Fore.GREEN}Training Complete!{Style.RESET_ALL}")
    print(f"Best accuracy achieved: {stats.best_acc * 100:.2f}%")
    print(f"Total training time: {stats.get_elapsed_time()}")

    # Handle model saving
    print(
        f"\n{Fore.YELLOW}Would you like to save the model? [Y/n]:{Style.RESET_ALL} ",
        end="",
    )
    if input().lower().strip() in ["", "y", "yes"]:
        save_path = save_model(model, stats)
        print(f"{Fore.GREEN}Model saved to: {save_path}{Style.RESET_ALL}")

    # Offer prediction on local images
    print(
        f"\n{Fore.YELLOW}Would you like to predict on local images? [Y/n]:{Style.RESET_ALL} ",
        end="",
    )
    if input().lower().strip() in ["", "y", "yes"]:
        predict_local_images(model)


# Entry point with cursor management safety
if __name__ == "__main__":
    try:
        main()
    finally:
        show_cursor()
