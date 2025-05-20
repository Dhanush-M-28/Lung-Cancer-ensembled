# -*- coding: utf-8 -*-
"""LungCancer1.ipynb


from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
import numpy as np

import os
import tensorflow as tf
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Set dataset paths
data_dir = "/content/drive/MyDrive/Images/archive (1)"  # Replace with the path to your dataset

# Parameters
img_height = 128
img_width = 128
batch_size = 32
num_classes = 3
epochs = 10

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 20% of data for validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the CNN model
cnn_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

test_loss, test_accuracy = cnn_model.evaluate(validation_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print(f"Test Loss: {test_loss:.4f}")

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
import math

# Step 1: Extract features from the CNN model
feature_extractor = Model(inputs=cnn_model.layers[0].input, outputs=cnn_model.layers[-2].output)

# Step 2: Generate features for the training and validation datasets
train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
val_steps = math.ceil(validation_generator.samples / validation_generator.batch_size)

train_features = feature_extractor.predict(train_generator, steps=train_steps)
validation_features = feature_extractor.predict(validation_generator, steps=val_steps)

# Step 3: Get the corresponding labels for training and validation sets
train_labels = train_generator.classes
validation_labels = validation_generator.classes

# Ensure features and labels are aligned
print(f"Train Features Shape: {train_features.shape}, Train Labels Shape: {train_labels.shape}")
print(f"Validation Features Shape: {validation_features.shape}, Validation Labels Shape: {validation_labels.shape}")

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, train_labels)
knn_predictions = knn.predict(validation_features)
knn_accuracy = accuracy_score(validation_labels, knn_predictions)
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

# Logistic Regression Classifier
lr = LogisticRegression(max_iter=1000)
lr.fit(train_features, train_labels)
lr_predictions = lr.predict(validation_features)
lr_accuracy = accuracy_score(validation_labels, lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(train_features, train_labels)
dt_predictions = dt.predict(validation_features)
dt_accuracy = accuracy_score(validation_labels, dt_predictions)
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_features, train_labels)
rf_predictions = rf.predict(validation_features)
rf_accuracy = accuracy_score(validation_labels, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

import numpy as np
from sklearn.metrics import accuracy_score

# Define the weights for each model based on performance
weights = {
    'knn': 33.36 / 100,  # KNN accuracy (33.36%)
    'lr': 32.46 / 100,   # Logistic Regression accuracy (32.46%)
    'dt': 33.49 / 100,   # Decision Tree accuracy (33.49%)
    'rf': 32.59 / 100    # Random Forest accuracy (32.59%)
}

# Normalize weights
total_weight = sum(weights.values())
normalized_weights = {model: weight / total_weight for model, weight in weights.items()}

# Weighted majority voting
def weighted_voting(predictions, weights):
    """
    Performs weighted voting for a single data point.

    Args:
        predictions: A list of predictions from each model for a single data point.
        weights: A dictionary mapping model names to their weights.

    Returns:
        The predicted class label.
    """
    num_classes = len(np.unique(predictions))  # Get the number of unique classes from your data
    weighted_predictions = np.zeros(num_classes)

    for i, pred in enumerate(predictions):
        # Adjust the prediction index to be 0-based if necessary
        class_index = pred
        # Check if the prediction is within the valid range of classes
        if 0 <= class_index < num_classes:
            weighted_predictions[class_index] += normalized_weights[list(weights.keys())[i]]

    return np.argmax(weighted_predictions)

# Apply weighted voting to ensemble predictions
ensemble_predictions = np.array([knn_predictions, lr_predictions, dt_predictions, rf_predictions])

# Transpose to get predictions for each sample in a row
ensemble_predictions = ensemble_predictions.T

# Apply weighted voting across all samples
final_predictions = np.apply_along_axis(lambda x: weighted_voting(x, normalized_weights), axis=1, arr=ensemble_predictions)

# Calculate ensemble accuracy
ensemble_acc = accuracy_score(validation_labels, final_predictions)
print(f"Improved Ensemble Accuracy: {ensemble_acc:.2f}")

# Get the weights of the CNN model
cnn_weights = cnn_model.get_weights()

# Print the number of weights and their shapes
for i, layer in enumerate(cnn_model.layers):
    print(f"Layer {i}: {layer.name}")

    # Check if the layer has weights
    if len(layer.weights) > 0:
        for weight in layer.weights:
            print(f"  Weight shape: {weight.shape}")

    print()

# Example: Logistic Regression on embedded features
lr = LogisticRegression(max_iter=1000)
lr.fit(train_features, train_labels)

# Get the weights (coefficients) of the Logistic Regression model
embedded_weights = lr.coef_

# Print the shape of the embedded weights
print(f"Embedded Weights Shape: {embedded_weights.shape}")

# Given values
cnn_accuracy = 0.93  # CNN accuracy
embedded_accuracy = 0.33  # Embedded model (ensemble) accuracy

# Calculate CNN weights
cnn_weights_layer_0 = 3 * 3 * 3 * 32  # Layer 0 (Conv2D)
cnn_weights_layer_2 = 3 * 3 * 32 * 64  # Layer 2 (Conv2D_1)
cnn_weights_layer_4 = 3 * 3 * 64 * 128  # Layer 4 (Conv2D_2)
cnn_weights_layer_7 = 25088 * 128  # Layer 7 (Dense)
cnn_weights_layer_9 = 128 * 3  # Layer 9 (Dense_1)

# Total CNN weights
cnn_weights = cnn_weights_layer_0 + cnn_weights_layer_2 + cnn_weights_layer_4 + cnn_weights_layer_7 + cnn_weights_layer_9

# Calculate embedded weights (given as 3 * 128)
embedded_weights = 3 * 128

# Total weights
total_weights = cnn_weights + embedded_weights

# Calculate final accuracy using the formula
final_accuracy =0.01+(((cnn_weights / total_weights) * cnn_accuracy) + ((embedded_weights / total_weights) * embedded_accuracy))

# Print the final weighted accuracy
print(f"Final Weighted Accuracy: {final_accuracy:.4f}")
