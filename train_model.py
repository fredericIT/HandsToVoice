#!/usr/bin/env python3
"""
HandsToVoice — Model Training Script
Trains a neural network model for Kinyarwanda Sign Language classification.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Using scikit-learn for model training.")
    TENSORFLOW_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

from src.vocabulary import Vocabulary


class KSLModelTrainer:
    """Trainer for Kinyarwanda Sign Language classification models."""

    def __init__(self, data_dir="data/processed", models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.vocabulary = Vocabulary()
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Training data
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.label_encoder = None
        self.num_classes = 0
        
        # Model
        self.model = None
        self.history = None

    def load_data(self, data_file=None):
        """Load training data from CSV file."""
        if data_file is None:
            # Look for the most recent data file
            data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if not data_files:
                raise FileNotFoundError(f"No data files found in {self.data_dir}")
            data_file = os.path.join(self.data_dir, sorted(data_files)[-1])
        
        print(f"[Training] Loading data from {data_file}")
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Check expected columns
        expected_columns = [f'landmark_{i}' for i in range(63)] + ['label']
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in data file: {missing_columns}")
        
        # Separate features and labels
        X = df[[f'landmark_{i}' for i in range(63)]].values
        y = df['label'].values
        
        print(f"[Training] Loaded {len(X)} samples")
        print(f"[Training] Unique labels: {len(np.unique(y))}")
        
        return X, y

    def preprocess_data(self, X, y, test_size=0.2, val_size=0.15):
        """Preprocess and split the data."""
        print("[Training] Preprocessing data...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"[Training] Number of classes: {self.num_classes}")
        print(f"[Training] Classes: {list(self.label_encoder.classes_)}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        # Normalize features (zero mean, unit variance)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Store data
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"[Training] Train set: {len(X_train)} samples")
        print(f"[Training] Validation set: {len(X_val)} samples")
        print(f"[Training] Test set: {len(X_test)} samples")
        
        # Save scaler for later use
        import joblib
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"[Training] Scaler saved to {scaler_path}")

    def create_model(self, model_type='neural_network'):
        """Create the classification model."""
        print(f"[Training] Creating {model_type} model...")
        
        if model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
            self.model = self._create_neural_network()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print("[Training] Model created successfully")

    def _create_neural_network(self):
        """Create a neural network model for sign classification."""
        model = keras.Sequential([
            layers.Input(shape=(63,)),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        return model

    def train_model(self, epochs=50, batch_size=32, verbose=1):
        """Train the model."""
        print("[Training] Starting model training...")
        
        if isinstance(self.model, keras.Model):
            # Neural network training
            self.history = self.model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
        else:
            # Scikit-learn model training
            self.model.fit(self.X_train, self.y_train)
            self.history = None
        
        print("[Training] Model training completed")

    def evaluate_model(self):
        """Evaluate the trained model."""
        print("[Training] Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        
        if isinstance(self.model, keras.Model):
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_pred_proba = np.max(y_pred, axis=1)
        else:
            y_pred_classes = y_pred
            y_pred_proba = self.model.predict_proba(self.X_test)
            y_pred_proba = np.max(y_pred_proba, axis=1)
        
        # Accuracy
        accuracy = np.mean(y_pred_classes == self.y_test)
        print(f"[Training] Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            self.y_test, y_pred_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        print("\n[Training] Classification Report:")
        print(classification_report(self.y_test, y_pred_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_classes)
        
        # Plot results
        self._plot_training_results()
        self._plot_confusion_matrix(cm, class_names)
        
        return accuracy, report, cm

    def _plot_training_results(self):
        """Plot training history for neural networks."""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.models_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Training] Training plots saved to {plot_path}")

    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plot_path = os.path.join(self.models_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Training] Confusion matrix saved to {plot_path}")

    def save_model(self, model_name='ksl_model'):
        """Save the trained model."""
        print("[Training] Saving model...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if isinstance(self.model, keras.Model):
            # Save Keras model
            model_path = os.path.join(self.models_dir, f'{model_name}.h5')
            self.model.save(model_path)
            print(f"[Training] Keras model saved to {model_path}")
        else:
            # Save scikit-learn model
            import joblib
            model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
            joblib.dump(self.model, model_path)
            print(f"[Training] Scikit-learn model saved to {model_path}")
        
        # Save label encoder
        labels_path = os.path.join(self.models_dir, f'{model_name}_labels.npy')
        np.save(labels_path, self.label_encoder.classes_)
        print(f"[Training] Labels saved to {labels_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'neural_network' if isinstance(self.model, keras.Model) else 'sklearn',
            'num_classes': self.num_classes,
            'feature_length': 63,
            'classes': list(self.label_encoder.classes_),
            'timestamp': timestamp,
            'test_accuracy': float(np.mean(self.model.predict(self.X_test) == self.y_test)) if hasattr(self.model, 'predict') else 0.0
        }
        
        metadata_path = os.path.join(self.models_dir, f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[Training] Metadata saved to {metadata_path}")

    def run_training(self, data_file=None, model_type='neural_network', epochs=50):
        """Run the complete training pipeline."""
        try:
            # Load data
            X, y = self.load_data(data_file)
            
            # Preprocess
            self.preprocess_data(X, y)
            
            # Create model
            self.create_model(model_type)
            
            # Train
            self.train_model(epochs=epochs)
            
            # Evaluate
            accuracy, report, cm = self.evaluate_model()
            
            # Save model
            self.save_model()
            
            print("[Training] Training completed successfully!")
            print(f"[Training] Final test accuracy: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            print(f"[Training] Error during training: {e}")
            raise


def create_sample_data():
    """Create sample training data for demonstration."""
    print("[Training] Creating sample data for demonstration...")
    
    # Create data directory
    data_dir = "data/processed"
    os.makedirs(data_dir, exist_ok=True)
    
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
    
    # Save to CSV
    data_file = os.path.join(data_dir, 'sample_data.csv')
    df.to_csv(data_file, index=False)
    print(f"[Training] Sample data saved to {data_file}")
    
    return data_file


def main():
    """Main training script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train KSL classification model")
    parser.add_argument("--data", type=str, help="Path to training data CSV file")
    parser.add_argument("--model-type", type=str, default="neural_network",
                       choices=["neural_network", "random_forest", "svm"],
                       help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for neural network")
    parser.add_argument("--sample-data", action="store_true", help="Create sample data for testing")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HandsToVoice - Model Training")
    print("University of Rwanda - College of Science and Technology")
    print("=" * 60)
    
    # Create sample data if requested
    if args.sample_data:
        data_file = create_sample_data()
        args.data = data_file
    
    # Check for data file
    if not args.data:
        # Look for existing data files
        data_dir = "data/processed"
        if os.path.exists(data_dir):
            data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if data_files:
                args.data = os.path.join(data_dir, sorted(data_files)[-1])
                print(f"[Training] Using existing data file: {args.data}")
            else:
                print("[Training] No data file found. Use --sample-data to create sample data.")
                return 1
        else:
            print("[Training] No data directory found. Use --sample-data to create sample data.")
            return 1
    
    # Run training
    trainer = KSLModelTrainer()
    
    try:
        accuracy = trainer.run_training(
            data_file=args.data,
            model_type=args.model_type,
            epochs=args.epochs
        )
        
        print("\n[Training] Training completed successfully!")
        print(f"[Training] Final accuracy: {accuracy:.4f}")
        print("[Training] Model saved to 'models/' directory")
        print("[Training] You can now run the main application: python main.py")
        
        return 0
        
    except Exception as e:
        print(f"[Training] Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
