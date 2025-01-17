"""Module for handling image loading and prediction functionality."""

# Standard library for finding files matching patterns (*.png, *.jpg)
import glob
import os

# OpenCV library for image processing (loading, resizing, thresholding)
import cv2

# NumPy for efficient array operations on image data
import numpy as np

# TinyGrad's Tensor for neural network operations
from tinygrad import Tensor

# Colorama for colored terminal output
from colorama import Fore, Style


def load_and_preprocess_image(image_path):
    """Load and preprocess a local image for MNIST prediction."""
    # Step 1: Load the image in grayscale mode (single channel, like MNIST)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Make sure the image was loaded successfully
    if img is None:
        raise ValueError("Could not load image")

    # Check if image is already in MNIST format (28x28 pixels)
    is_mnist = img.shape[0] == 28 and img.shape[1] == 28

    if not is_mnist:
        # Step 2: Resize and preprocess non-MNIST images
        # Start with a larger size for better quality processing
        target_size = 128
        aspect_ratio = img.shape[1] / img.shape[0]
        if aspect_ratio > 1:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Step 3: Enhance image contrast to separate digit from background
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Step 4: Convert to pure black and white using adaptive thresholding
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 5: Find and crop to the digit's bounding box
        coords = cv2.findNonZero(255 - img)
        if coords is not None:
            # Get the rectangle containing the digit
            x, y, w, h = cv2.boundingRect(coords)
            # Add 20% padding around the digit
            padding = int(max(w, h) * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            img = img[y : y + h, x : x + w]

        # Step 6: Make the image square by adding black padding
        size = max(img.shape)
        square = np.zeros((size, size), dtype=np.uint8)
        offset_y = (size - img.shape[0]) // 2
        offset_x = (size - img.shape[1]) // 2
        square[
            offset_y : offset_y + img.shape[0], offset_x : offset_x + img.shape[1]
        ] = img

        # Step 7: Resize to MNIST size (28x28 pixels)
        img = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # Step 8: Convert to float32 (keep 0-255 range like MNIST)
    img = img.astype(np.float32)

    # Step 9: Invert if needed (MNIST uses white digits on black background)
    if np.mean(img) > 127:
        img = 255 - img

    # Step 10: Reshape to match MNIST format: (batch_size=1, channels=1, height=28, width=28)
    img = img.reshape(1, 1, 28, 28)

    # Step 11: Convert to TinyGrad tensor for model input
    return Tensor(img)


def predict_local_images(model):
    """Predict digits from local PNG and JPG files in the pics directory."""
    # Define the pics directory path
    pics_dir = "pics"

    # Create pics directory if it doesn't exist
    if not os.path.exists(pics_dir):
        os.makedirs(pics_dir)
        print(
            f"{Fore.YELLOW}Created 'pics' directory for storing images.{Style.RESET_ALL}"
        )

    while True:
        # Find all PNG and JPG files in the pics directory
        image_files = glob.glob(os.path.join(pics_dir, "*.png")) + glob.glob(
            os.path.join(pics_dir, "*.jpg")
        )
        image_files.sort()  # Sort for consistent ordering

        # Exit if no images found
        if not image_files:
            print(
                f"{Fore.YELLOW}No PNG or JPG files found in the '{pics_dir}' directory.{Style.RESET_ALL}"
            )
            return

        # Display list of found images (show just filenames, not full paths)
        print(f"\n{Fore.GREEN}Found {len(image_files)} image(s):{Style.RESET_ALL}")
        for i, image_path in enumerate(image_files, 1):
            filename = os.path.basename(image_path)
            print(f"{i}. {filename}")

        # Show user options
        print(f"\n{Fore.CYAN}Options:{Style.RESET_ALL}")
        print("1-N. Select image number to predict")
        print("a. Predict all images")
        print("q. Quit")

        # Get user choice
        choice = (
            input(f"\n{Fore.CYAN}Enter your choice:{Style.RESET_ALL} ").strip().lower()
        )

        if choice == "q":
            print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")
            return
        elif choice == "a":
            # Process all images in batch
            print(f"\n{Fore.GREEN}Processing all images...{Style.RESET_ALL}")
            for image_path in image_files:
                try:
                    # Load and preprocess the image
                    img_tensor = load_and_preprocess_image(image_path)
                    # Set model to evaluation mode (disables dropout)
                    Tensor.training = False
                    # Get prediction (digit with highest probability)
                    prediction = model(img_tensor).argmax().item()
                    # Calculate confidence using softmax
                    confidence = model(img_tensor).softmax()[0][prediction].item()
                    filename = os.path.basename(image_path)
                    print(
                        f"{Fore.CYAN}{filename}:{Style.RESET_ALL} Predicted digit: {prediction} (Confidence: {confidence:.2f})"
                    )
                except Exception as e:
                    filename = os.path.basename(image_path)
                    print(
                        f"{Fore.RED}Error processing {filename}: {str(e)}{Style.RESET_ALL}"
                    )
        else:
            # Process single selected image
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(image_files):
                    image_path = image_files[idx]
                    try:
                        # Load and preprocess the image
                        img_tensor = load_and_preprocess_image(image_path)
                        # Set model to evaluation mode
                        Tensor.training = False
                        # Get prediction and confidence
                        prediction = model(img_tensor).argmax().item()
                        confidence = model(img_tensor).softmax()[0][prediction].item()
                        filename = os.path.basename(image_path)
                        print(
                            f"\n{Fore.CYAN}{filename}:{Style.RESET_ALL} Predicted digit: {prediction} (Confidence: {confidence:.2f})"
                        )
                    except Exception as e:
                        filename = os.path.basename(image_path)
                        print(
                            f"{Fore.RED}Error processing {filename}: {str(e)}{Style.RESET_ALL}"
                        )
                else:
                    print(
                        f"{Fore.RED}Invalid image number. Please try again.{Style.RESET_ALL}"
                    )
            except ValueError:
                print(f"{Fore.RED}Invalid input. Please try again.{Style.RESET_ALL}")

        # Ask if user wants to continue
        print("\n" + "=" * 60)  # Separator line
        choice = (
            input(f"{Fore.YELLOW}Continue predicting? [Y/n]:{Style.RESET_ALL} ")
            .lower()
            .strip()
        )
        if choice not in ["", "y", "yes"]:
            print(f"{Fore.GREEN}Goodbye!{Style.RESET_ALL}")
            return
