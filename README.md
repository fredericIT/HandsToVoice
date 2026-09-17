# HandsToVoice

A Real-Time Kinyarwanda Sign Language Recognition and Voice Conversion System

**University of Rwanda - College of Science and Technology**  
**Department of Information and Communication Technology**

## Overview

HandsToVoice is an assistive technology system that bridges the communication gap between people with speech disabilities and the general public. The system uses a standard PC webcam to detect and recognize Kinyarwanda Sign Language (KSL) gestures in real time, then converts those signs into spoken Kinyarwanda words through the computer's speaker.

## Features

- 🎥 **Real-time hand gesture recognition** using MediaPipe and OpenCV
- 🧠 **Machine learning classification** with TensorFlow/Keras neural networks
- 🔊 **Kinyarwanda text-to-speech** conversion (online and offline options)
- 🖥️ **Modern PyQt5 GUI** with dark theme interface
- 📊 **Data collection tools** for training custom models
- 🎯 **10 basic KSL signs** included (expandable vocabulary)

## System Requirements

- Python 3.8+
- PC webcam (standard USB or built-in)
- Speakers or headphones
- Internet connection (for online TTS, optional)

## Installation

1. **Clone or download the project:**
   ```bash
   cd HandsToVoice
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Option 1: Use Sample Data (Demo)

1. **Train a model with sample data:**
   ```bash
   python train_model.py --sample-data --epochs 20
   ```

2. **Run the main application:**
   ```bash
   python main.py
   ```

### Option 2: Collect Your Own Data

1. **Collect training data:**
   ```bash
   python collect_data.py
   ```
   - Follow the GUI instructions
   - Collect samples for each sign
   - Save data files

2. **Combine and train:**
   ```bash
   # First, merge your collected data files
   python train_model.py --data path/to/your/data.csv
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Usage Guide

### Main Application

1. **Start Recognition:** Click "▶️ Start Recognition" to begin
2. **Perform Signs:** Make KSL gestures in front of the camera
3. **View Results:** See recognized signs and confidence levels
4. **Build Sentences:** Signs automatically form sentences
5. **Voice Output:** Click "🔊 Speak" to hear the sentence

### Data Collection

1. **Select Sign:** Choose which KSL sign to collect
2. **Set Target:** Choose number of samples (30-50 recommended)
3. **Start Collection:** Click "▶️ Start Collection"
4. **Perform Sign:** Make the gesture consistently
5. **Save Data:** Click "💾 Save Data" when complete

### Model Training

```bash
# Train with existing data
python train_model.py --data data/processed/your_data.csv

# Train with sample data
python train_model.py --sample-data

# Use different model types
python train_model.py --model-type random_forest
python train_model.py --model-type svm
```

## Project Structure

```
HandsToVoice/
├── main.py                 # Main application entry point
├── train_model.py          # Model training script
├── collect_data.py         # Data collection GUI
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Source modules
│   ├── capture.py         # Camera capture
│   ├── detector.py        # Hand landmark detection
│   ├── classifier.py      # Sign classification
│   ├── tts.py            # Text-to-speech
│   ├── vocabulary.py      # Sign vocabulary
│   └── gui/              # GUI components
│       ├── main_window.py
│       ├── camera_widget.py
│       └── styles.py
├── data/                  # Data files
│   ├── labels.json       # Vocabulary definitions
│   ├── raw/              # Collected raw data
│   └── processed/        # Processed training data
├── models/               # Trained models
└── assets/               # Additional resources
```

## Supported KSL Signs

The system includes 10 basic KSL signs:

| Sign | Kinyarwanda | English | Category |
|------|------------|---------|----------|
| muraho | Muraho | Hello | Greetings |
| amakuru | Amakuru | How are you? | Greetings |
| yego | Yego | Yes | Common |
| oya | Oya | No | Common |
| murakoze | Murakoze | Thank you | Greetings |
| mbabarira | Mbabarira | Sorry/Excuse me | Common |
| ndagukunda | Ndagukunda | I love you | Common |
| amazi | Amazi | Water | Emergency |
| ubufasha | Ubufasha | Help | Emergency |
| muganga | Muganga | Doctor | Emergency |

## Command Line Options

### Main Application
```bash
python main.py --camera 0 --offline-tts
```

### Data Collection
```bash
python collect_data.py --camera 0 --output data/raw
```

### Model Training
```bash
python train_model.py --data data.csv --model-type neural_network --epochs 50
```

## Model Performance

- **Target Accuracy:** 85%+ on test dataset
- **Processing Speed:** 15+ FPS on standard laptop
- **Vocabulary Size:** 10 signs (expandable)
- **Feature Length:** 63 hand landmarks (21 points × 3 coordinates)

## Troubleshooting

### Camera Issues
- Check camera connection and permissions
- Try different camera index: `python main.py --camera 1`
- Ensure no other app is using the camera

### Model Not Loading
- Train a model first: `python train_model.py --sample-data`
- Check model files in `models/` directory

### TTS Issues
- **Online TTS:** Requires internet connection
- **Offline TTS:** Use `--offline-tts` flag
- Check system audio output

### GUI Display Issues
- Install Qt platform plugins: `sudo apt install python3-pyqt5`
- Use display server: `export QT_QPA_PLATFORM=xcb`

## Development

### Adding New Signs

1. **Update vocabulary** in `data/labels.json`
2. **Collect training data** using `collect_data.py`
3. **Retrain model** using `train_model.py`
4. **Test with main application**

### Model Architecture

The neural network uses:
- Input: 63 hand landmark features
- Hidden layers: 128 → 64 → 32 neurons
- Dropout: 30% regularization
- Output: Softmax classification
- Optimizer: Adam
- Loss: Sparse categorical crossentropy

### Data Pipeline

1. **Capture:** Webcam frames at 30 FPS
2. **Detection:** MediaPipe hand landmark extraction
3. **Normalization:** Position-invariant feature vectors
4. **Classification:** Trained neural network
5. **Mapping:** Sign label to Kinyarwanda text
6. **Speech:** Text-to-speech conversion

## Contributing

This is an academic project for the University of Rwanda. Contributions welcome:

- **More KSL signs:** Expand the vocabulary
- **Performance optimization:** Improve speed and accuracy
- **Mobile version:** Android/iOS adaptation
- **Offline models:** Quantized models for edge devices

## License

This project is developed for educational and research purposes at the University of Rwanda.

## Acknowledgments

- **MediaPipe** by Google for hand tracking
- **TensorFlow** for machine learning
- **OpenCV** for computer vision
- **PyQt5** for GUI framework
- **University of Rwanda** for project support

## Contact

**Project:** HandsToVoice  
**Institution:** University of Rwanda - College of Science and Technology  
**Department:** Information and Communication Technology  

---

*Empowering communication through assistive technology*
