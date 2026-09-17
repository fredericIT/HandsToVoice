#!/usr/bin/env python3
"""
Simple training script without GUI components.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def create_sample_data():
    """Create sample training data."""
    print("Creating sample data...")
    
    # Generate sample landmarks for 10 signs
    np.random.seed(42)
    num_samples_per_sign = 100
    signs = ['muraho', 'amakuru', 'yego', 'oya', 'murakoze', 
             'mbabarira', 'ndagukunda', 'amazi', 'ubufasha', 'muganga']
    
    data = []
    for sign in signs:
        for _ in range(num_samples_per_sign):
            # Generate random landmarks (63 features)
            landmarks = np.random.randn(63) * 0.1
            
            # Add some sign-specific patterns
            sign_id = signs.index(sign)
            landmarks[sign_id % 63] += np.random.randn() * 0.5
            
            row = dict(zip([f'landmark_{i}' for i in range(63)], landmarks))
            row['label'] = sign
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def train_and_save_model():
    """Train and save a simple model."""
    print("Starting simple training...")
    
    # Create sample data
    df = create_sample_data()
    
    # Separate features and labels
    X = df[[f'landmark_{i}' for i in range(63)]].values
    y = df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"Training with {len(X)} samples, {num_classes} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create model
    model = keras.Sequential([
        layers.Input(shape=(63,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=20, batch_size=32, verbose=1)
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/ksl_model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save labels
    labels_path = 'models/ksl_model_labels.npy'
    np.save(labels_path, label_encoder.classes_)
    print(f"Labels saved to {labels_path}")
    
    # Save scaler
    import joblib
    scaler_path = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'neural_network',
        'num_classes': num_classes,
        'feature_length': 63,
        'classes': list(label_encoder.classes_),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'test_accuracy': float(accuracy)
    }
    
    import json
    metadata_path = 'models/ksl_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    return accuracy


if __name__ == "__main__":
    print("=" * 50)
    print("HandsToVoice - Simple Model Training")
    print("=" * 50)
    
    accuracy = train_and_save_model()
    
    print("\nTraining completed successfully!")
    print(f"Final accuracy: {accuracy:.4f}")
    print("Model files saved in 'models/' directory")
    print("You can now run: python main.py")
