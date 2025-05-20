# Lung-Cancer-ensembled
A hydrid approach of machine learning and deep learning models
This project aims to detect and classify lung cancer images using a hybrid approach: a Convolutional Neural Network (CNN) for image classification and feature extraction, followed by traditional Machine Learning models (like KNN, Logistic Regression, Decision Tree, and Random Forest) for further classification and ensemble learning.

The dataset used consists of medical images of lungs, categorized into three classes namely Adenocarcinoma, Benign, and SquamousCell Carcinoma.

Technologies & Libraries Used :
Python 3
TensorFlow / Keraspr
Scikit-learn
NumPy
Google Colab

Project Workflow :
Data Preprocessing: Resize, normalize, and augment lung images using ImageDataGenerator with an 80/20 train-validation split.
CNN Training: Build and train a CNN model for multi-class image classification.
Feature Extraction: Use the trained CNN to extract features (embeddings) from the penultimate dense layer.
Classical ML Models: Train KNN, Logistic Regression, Decision Tree, and Random Forest on the extracted features.
Ensemble Learning: Combine predictions from classical models using weighted majority voting based on individual model accuracies.
Final Accuracy Calculation: Compute a weighted final accuracy by combining CNN and ensemble performance based on model weight counts.
