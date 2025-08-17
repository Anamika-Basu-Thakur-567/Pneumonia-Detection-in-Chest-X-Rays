# Pneumonia-Detection-in-Chest-X-Rays

## Project Overview
This project implements a deep learning pipeline to diagnose pneumonia from chest X-ray images, a critical task in medical imaging for accelerating clinical decision-making. The goal is to build a robust binary classification model that can accurately distinguish between "Normal" and "Pneumonia" cases. The notebook covers the complete machine learning workflow, starting from data acquisition and strategic preprocessing to address class imbalance, followed by the development and training of multiple Convolutional Neural Network (CNN) architectures. The final model leverages Transfer Learning with a ResNet50 backbone, fine-tuned to achieve high diagnostic accuracy, demonstrating a practical application of computer vision in healthcare.

## Key Features
- **Data Handling & Preprocessing:**
   - Downloads and extracts the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle.
   - Performs a custom 70/20/10 train-validation-test split to create a more robust evaluation set than the original dataset provided.
   - Utilizes Albumentations for advanced image augmentation, applying stronger transformations to the under-represented "Normal" class to combat class imbalance.

- **Model Development & Training:**
   - Builds a baseline custom CNN using TensorFlow/Keras to establish initial performance.
   - Develops an improved CNN by incorporating modern techniques like Batch Normalization, Dropout, and L2 Regularization to enhance training stability and reduce overfitting.
   - Implements a state-of-the-art Transfer Learning model using a pre-trained ResNet50 architecture, followed by a fine-tuning stage for optimal performance.

- **Comprehensive Evaluation:**
   - Tracks training and validation accuracy/loss across epochs to monitor model learning.
   - Evaluates the final model on the unseen test set to provide an unbiased measure of performance.
   - Generates a Confusion Matrix to analyze class-specific performance, focusing on minimizing false negatives.
   - Calculates the ROC AUC Score to assess the model's overall discriminative ability.


