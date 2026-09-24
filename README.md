# HandsToVoice

A real-time Kinyarwanda Sign Language recognition and voice conversion system.

**University of Rwanda — College of Science and Technology**
**Department of Information and Communication Technology**

## Overview

HandsToVoice helps a signer and a hearing person talk to each other, both ways, using an ordinary laptop:

- **Sign → voice:** the signer signs in front of the webcam. The system recognizes each sign and speaks the Kinyarwanda word aloud.
- **Voice → sign:** a hearing person speaks Kinyarwanda. When the speech contains a vocabulary word, the system shows the video of that sign to the signer.

The system recognizes signs one at a time from a vocabulary of 36 signs (see [Vocabulary](#vocabulary)).

## Features

- **Sign recognition.** MediaPipe tracks 21 hand points and the face in each webcam frame. An LSTM network classifies each 30-frame sequence. Features are made independent of hand position and camera distance, and include the hand's position relative to the face.
- **Batch speaking with undo.** Recognized signs are collected in groups of 3 and spoken after a short delay. During that delay, "Undo Last Sign" can cancel a wrong detection before anyone hears it.
- **Recorded Kinyarwanda voices.** Each sign plays a real recorded voice. If a sign has no recording, the computer's built-in text-to-speech is used instead.
- **Voice listening.** The system picks out vocabulary words from natural spoken sentences and shows each sign's video in a pop-up. Recognition uses Google's speech service when online and Meta's MMS model, running on the laptop, when offline.
- **Sign management in the app.** Add a sign (define, record, train, test), edit or delete signs, manage voice recordings, and practise with "Learn a Sign".
- **Low-light warning** when the room is too dark for reliable hand tracking.

## Requirements

- Linux; developed and tested on Ubuntu 24.04. The microphone is read through `arecord` (`sudo apt install alsa-utils`).
- Python 3.12.
- A webcam, a microphone, and speakers.
- About 4 GB of free disk space for the offline speech model. It downloads automatically the first time listening starts.
- Internet is optional. Without it, listening uses the offline model only.

## Installation

```bash
cd HandsToVoice
python3 -m venv venv
venv/bin/python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
venv/bin/python3 -m pip install -r requirements.txt
```

Installing the CPU build of PyTorch first avoids downloading the much larger GPU build.

## Running

```bash
./run.sh                  # or: ./run.sh --camera 1
./install_launcher.sh     # optional: adds HandsToVoice to the applications menu
```

In the app:

1. Click **▶️ Start Recognition** and sign in front of the camera.
2. After 3 signs, the sentence is spoken automatically. Use **Undo Last Sign** to cancel a wrong one, or **🔊 Speak** to speak now.
3. **🎧 Listening** is on by default. When a hearing person speaks, the matching sign videos pop up. Click the button to turn listening off.

## Adding a new sign

Use **➕ Add Sign** in the app. The wizard records the sign (several takes; more takes and small variations give better accuracy), extracts the hand landmarks, retrains the model and lets you test it straight away. Add the sign's Kinyarwanda voice recording in the same wizard or later in **Manage Voices**.

## Evaluation

```bash
venv/bin/python3 evaluate_model.py
```

This runs 5-fold cross-validation with the app's own training pipeline (`src/training.py`), so every recording is predicted by a model that never saw it. It writes:

- `reports/evaluation_report.md` — overall and per-sign accuracy, the most confused sign pairs, accuracy when the face is not detected, and prediction latency
- `reports/confusion_matrix.png`
- `reports/evaluation_results.json`

The deployed model in `models/` is not changed.

## Tests

```bash
venv/bin/python3 -m unittest discover -s tests -t .
```

These cover landmark normalization, the training data split and augmentation, how speech is cut into utterances, how spoken words are matched to signs, and the online speech time-out and fallback. They need no camera, microphone or internet.

## Privacy

- While listening is on, each spoken phrase is sent to Google's speech service when the computer is online. The app shows a notice in the status bar while this happens. Offline, speech is recognized on the laptop and nothing leaves it.
- HandsToVoice never saves microphone audio or camera video during normal use. The only files it saves are the sign videos and voices you record on purpose in Add Sign, Edit Sign and Manage Voices.

## Limitations

- The recordings in this dataset come from one signer. Accuracy for a different person has not been measured yet; that needs recordings from additional signers kept out of training.
- The system recognizes isolated signs, not continuous signed sentences. Sign language has its own grammar, which is not translated.
- The vocabulary is 36 signs.

## Project structure

```
HandsToVoice/
├── main.py                  # Application entry point
├── run.sh                   # Launcher
├── install_launcher.sh      # Adds an applications-menu entry
├── evaluate_model.py        # Cross-validated evaluation report
├── requirements.txt
├── src/
│   ├── capture.py           # Webcam capture
│   ├── detector.py          # MediaPipe hand and face detection
│   ├── normalize.py         # Landmark normalization (65 features per frame)
│   ├── classifier.py        # LSTM sign classifier (inference)
│   ├── training.py          # Training pipeline shared by the app and evaluation
│   ├── tts.py               # Voice output (recordings + text-to-speech fallback)
│   ├── speech_listener.py   # Microphone listening and speech segmentation
│   ├── speech_recognizer.py # Speech-to-text (online/offline) and word matching
│   ├── lighting.py          # Low-light detection
│   ├── vocabulary.py        # Sign vocabulary (data/labels.json)
│   └── gui/                 # PyQt5 windows and dialogs
├── tests/                   # Automated tests
├── data/
│   ├── labels.json          # Vocabulary definitions
│   ├── videos/              # Recorded sign videos (one folder per sign)
│   ├── sequences/           # Extracted landmark sequences used for training
│   └── audio/               # Recorded Kinyarwanda voices
├── models/                  # Trained model, labels and metadata
└── reports/                 # Output of evaluate_model.py
```

Older scripts in the project root (`train_model.py`, `train_simple.py`, `collect_data.py`, `add_amaso_and_retrain.py`, `add_letter_c_and_retrain.py`) belong to an earlier frame-by-frame approach and are not used by the app.

## Vocabulary

| Kinyarwanda | English |
|---|---|
| muraho | hello |
| umeze gute ?? | how are you? |
| ameze | she / he is |
| neza | well |
| yego | yes |
| oya | no |
| nyabuneka | please |
| mbabarira | sorry |
| ndagukunda | I love you |
| itonde | be careful |
| tangira | start |
| witeguye | witeguye |
| birasobanutse | birasobanutse |
| gusangiza | to share |
| tugiye | we are going |
| atandukanye | different |
| benshi | benshi |
| mwiza | mwiza |
| ni | is |
| ngewe | my |
| wowe | you |
| amazina yange | my name |
| mama | mother |
| papa | father |
| marume | uncle |
| masenge | aunt |
| abana | children |
| umunsi | day |
| uyu munsi | today |
| kuwa mbere | Monday |
| kuwa kabiri | Tuesday |
| kuwa gatatu | Wednesday |
| kuwa kane | Thursday |
| kuwa gatanu | Friday |
| kuwa gatandatu | Saturday |
| kucyumweru | Sunday |

## Acknowledgments

MediaPipe (Google) for hand and face tracking; TensorFlow for the sign model; Meta's MMS model and Hugging Face Transformers for offline speech recognition; OpenCV; PyQt5; and the University of Rwanda.
