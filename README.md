# Handwriting_digit_recognizer.py
Use of MNIST dataset to train a CNN network.  
# Handwritten Digit Recognizer (MNIST + CNN)

This repository contains a simple Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset.  
It’s designed as an educational project for learning image processing and deep learning fundamentals.

# Repository Visibility
Public:Anyone can view,fork,and contribute.
Editable via Pull Requests:Contributions are welcome through issues and PRs.

# Model Architecture
Conv2D (32 filters, 3×3, ReLU)
MaxPooling2D (2×2)
Conv2D (64 filters, 3×3, ReLU)
MaxPooling2D (2×2)
Flatten
Dense (64 units, ReLU)
