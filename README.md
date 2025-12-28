# Fashion-MNIST Basic Neural Network

## Project Overview

Built a beginner-friendly neural network project using the Fashion-MNIST
dataset to understand how basic neural networks work.\
The project focuses on learning core deep learning concepts such as data
normalization, neural network architecture, training, and evaluation
through step-by-step implementation and explanations, rather than
optimization or advanced mathematics.

## Table of Contents

-   Installation\
-   Dataset\
-   Preprocessing\
-   Model Training\
-   Evaluation\
-   Results\
-   How to Run\
-   Future Work

## Installation

To run this project, ensure Python is installed and install the required
dependencies:

``` bash
pip install tensorflow numpy matplotlib
```

## Dataset

The project uses the **Fashion-MNIST** dataset provided by
TensorFlow/Keras.

-   70,000 grayscale images of clothing\
-   Image size: 28 × 28 pixels\
-   10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat,
    Sandal, Shirt, Sneaker, Bag, Ankle boot)

The dataset is automatically downloaded when the script is executed.

## Preprocessing

Before training, the image data is preprocessed by:

-   Normalizing pixel values from the range **0--255** to **0--1**
-   Converting 2D images into 1D vectors using a Flatten layer

This preprocessing helps the neural network learn efficiently and
stably.

## Model Training

A simple **Multilayer Perceptron (MLP)** is used:

-   Flatten input layer
-   One hidden dense layer with ReLU activation
-   Output layer with Softmax activation

The model is trained using:

-   Adam optimizer
-   Sparse categorical cross-entropy loss
-   Accuracy as the evaluation metric

## Evaluation

After training, the model is evaluated on unseen test data to measure
its generalization performance.\
Accuracy is used to determine how well the model classifies clothing
images it has never seen before.

## Results

The model achieves reasonable accuracy for a basic neural network,
demonstrating that even a simple architecture can effectively learn
patterns from image data.

This confirms the effectiveness of neural networks for image
classification tasks.

## How to Run

1.  Clone the repository:

    ``` bash
    git clone <repository-url>
    ```

2.  Navigate to the project directory:

    ``` bash
    cd fashion-mnist-basic-neural-network
    ```

3.  Run the Python script:

    ``` bash
    python main.py
    ```

## Future Work

-   Implement the neural network using only NumPy (from scratch)
-   Add visualization of learned weights
-   Extend the project with Convolutional Neural Networks (CNNs)
-   Improve performance through hyperparameter tuning
-   Add detailed concept-wise notebooks for learning
