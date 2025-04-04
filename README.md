# MalarX-ServerSide

Server side of the deep learning-based image classification project to detect and classify malaria-infected blood cells. It detects 2 types of malaria:

* Plasmodium Falciparum

* Plasmodium Vivax

Built using TensorFlow/Keras, trained on microscopic cell images.

## Project Structure

malaria-classifier/
├── main.py                 # Server script
├── model.py                # Training script
├── malaria_model.h5        # Trained model 
├── README.md               # You're here
├── LICENSE

## Features

* Custom CNN architecture using Keras

* Multiclass classification (3 categories)

* High accuracy (up to 99–100% on validation)

* Real-time image prediction

## Installation

`pip install numpy opencv-python pillow matplotlib scikit-learn tensorflow imagecodecs`

Make sure you're using Python 3.10 or 3.11

## Required Imports

```import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout, BatchNormalization
)
```

## Training the Model

Run the training script:

`python model.py`
*IMPORTANT*: Uncomment the `if __name__ == 'main'`, or it won't run.

This will load and preprocess all images, train the CNN on 80% of data evaluate on the remaining 20%, plot accuracy/loss graphs, and save the model as malaria_model.h5.
Below is an example of an expected output from the training script.

![image](https://github.com/user-attachments/assets/affccddd-6347-404d-bd70-1ff3b1e8a531)

## Running the Server

Run the main script:

`python main.py`

This initiates the actual server, allowing the client to access it and function.

## Dataset Source

This project uses cell images from the NIH Malaria Dataset
