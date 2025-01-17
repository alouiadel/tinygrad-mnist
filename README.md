# TinyGrad MNIST Classifier 🔢 ✍️

Simple MNIST digit classifier using TinyGrad with clean terminal UI. Supports training, model saving/loading, and prediction on custom images.

## Quick Start 🚀
```bash
# Setup
python -m venv tiny-venv
source tiny-venv/bin/activate  # or `tiny-venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Train
python main.py

# Predict
# Place digit images (PNG/JPG) in the 'pics' folder
python main.py  # Select option to load model and predict
```

## Features ✨
- 🎯 Train on MNIST dataset with live progress display
- 💾 Save/load trained models
- 🔍 Predict digits from custom images
- 🎨 Clean terminal UI with colored output

Tinygrad documentation(initial source): [Training an MNIST Classifier](https://docs.tinygrad.org/mnist/).
