#!/usr/bin/env python3
"""
Model Loading Example for Breathing Irregularity Classification
===============================================================

This script demonstrates how to load and use the trained models
for breathing irregularity classification.

Generated on: 2025-07-13T19:45:27.867144
"""

import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

# Load the trained models
def load_models():
    """Load the trained models and preprocessing components."""
    
    # Load models
    cnn_model = tf.keras.models.load_model('path/to/cnn/model.h5')
    lstm_model = tf.keras.models.load_model('path/to/lstm/model.h5')
    
    # Load label encoder
    with open('../models/label_encoder_20250713_194527.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    return cnn_model, lstm_model, label_encoder

# Example prediction function
def predict_breathing_irregularity(signal_window, model, label_encoder):
    """
    Predict breathing irregularity for a 30-second signal window.
    
    Args:
        signal_window: numpy array of shape (960,) - 30 seconds at 32 Hz
        model: trained Keras model
        label_encoder: fitted LabelEncoder
    
    Returns:
        prediction: predicted class name
        confidence: prediction confidence (max probability)
    """
    
    # Reshape for model input (1, 960, 1)
    signal_reshaped = signal_window.reshape(1, -1, 1)
    
    # Get prediction probabilities
    probabilities = model.predict(signal_reshaped, verbose=0)
    
    # Get predicted class
    predicted_class_idx = np.argmax(probabilities[0])
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = probabilities[0][predicted_class_idx]
    
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    # Load models
    cnn_model, lstm_model, label_encoder = load_models()
    
    print("✅ Models loaded successfully!")
    print(f"🏷️  Available classes: {label_encoder.classes_}")
    
    # Example with random data (replace with real signal)
    example_signal = np.random.randn(960)  # 30 seconds at 32 Hz
    
    # Get predictions from both models
    cnn_pred, cnn_conf = predict_breathing_irregularity(example_signal, cnn_model, label_encoder)
    lstm_pred, lstm_conf = predict_breathing_irregularity(example_signal, lstm_model, label_encoder)
    
    print(f"\n📊 Predictions:")
    print(f"   1D CNN: {cnn_pred} (confidence: {cnn_conf:.3f})")
    print(f"   Conv-LSTM: {lstm_pred} (confidence: {lstm_conf:.3f})")
