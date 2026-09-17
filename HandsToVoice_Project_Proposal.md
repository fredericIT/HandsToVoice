# HandsToVoice — Project Proposal

---

## 1. Basic Information

| Field | Details |
|-------|---------|
| **Project Name** | **HandsToVoice** — Real-time Kinyarwanda Sign Language Recognition & Voice Conversion System |
| **Institution** | University of Rwanda — College of Science and Technology |
| **Department** | Information and Communication Technology (ICT) |
| **Date** | April 30, 2026 |

### Team Members

| Role | Full Name | Student ID |
|------|-----------|------------|
| **Team Lead** | NTAWUKURIRYAYO Frederic | 26-BK-RE-076 |
| **Team Member** | NYIRABACUMBITSI Melanie Celine | 26-BK-RE-055 |

---

## 2. Problem Statement

### The Communication Barrier

In Rwanda, an estimated **72,000+ people** live with hearing and speech disabilities (NISR, 2023). These individuals rely on **Kinyarwanda Sign Language (KSL)** to communicate, yet the vast majority of Rwandans — including teachers, healthcare providers, and service workers — do not understand sign language.

### Who Is Affected?

- **People with speech/hearing disabilities**: Cannot effectively communicate their needs, seek help, or participate in education and employment.
- **Healthcare professionals**: Struggle to diagnose and treat patients who rely on sign language, leading to misdiagnosis and delayed care.
- **Teachers and educators**: Cannot provide inclusive education to students with hearing impairments in mainstream classrooms.
- **Families and communities**: Experience social isolation when family members cannot communicate across the sign/spoken language divide.
- **Government and service providers**: Lack accessible tools to deliver public services to citizens with hearing disabilities.

### Current Gaps

- **No existing technological solution** for Kinyarwanda Sign Language recognition in Rwanda.
- Professional sign language interpreters are **scarce and expensive** — Rwanda has fewer than 100 certified KSL interpreters nationwide.
- Existing global sign language recognition tools (for ASL, BSL, etc.) do **not support Kinyarwanda** and are not adapted to local culture and gestures.

---

## 3. Proposed Solution

**HandsToVoice** is an AI-powered desktop application that uses a standard webcam to recognize Kinyarwanda Sign Language gestures in real time and converts them into spoken Kinyarwanda using text-to-speech technology.

### How It Works

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
│  Webcam  │───▶│  MediaPipe   │───▶│  Neural Net  │───▶│  Text    │───▶│  Voice   │
│  Camera  │    │  Hand        │    │  Classifier  │    │  Mapping │    │  Output  │
│          │    │  Landmark    │    │  (12 signs)  │    │  (KSL →  │    │  (gTTS / │
│          │    │  Detection   │    │              │    │  Kiny.)  │    │  pyttsx3)│
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘    └──────────┘
```

1. **Camera captures** the user performing a sign language gesture at 30 FPS.
2. **MediaPipe Hand Landmarker** extracts 21 hand landmark points (63 features: x, y, z per joint).
3. A **trained neural network** classifies the hand pose into one of the known KSL signs.
4. The recognized sign is **mapped to Kinyarwanda text** using the built-in vocabulary.
5. The Kinyarwanda text is **spoken aloud** through text-to-speech, enabling hearing people to understand the signer.

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Computer Vision | MediaPipe, OpenCV | Hand landmark detection |
| Machine Learning | TensorFlow / Keras | Sign classification |
| GUI Framework | PyQt5 | Desktop application interface |
| Text-to-Speech | gTTS (online) / pyttsx3 (offline) | Kinyarwanda voice output |
| Data Format | NumPy, Pandas, CSV | Landmark data processing |
| Language | Python 3.12 | Core programming language |

---

## 4. Key Features

### Feature 1: Real-Time Hand Detection & Tracking
Uses Google's MediaPipe Hand Landmarker to detect and track hand landmarks at 30 FPS with high accuracy. Supports detection of up to 2 hands simultaneously with position-invariant normalization.

### Feature 2: AI-Powered Sign Classification
A deep neural network (128→64→32 neurons with dropout and batch normalization) classifies hand poses into 12 KSL signs. The model uses a consensus-based approach — requiring 3 consecutive matching frames before confirming a sign — to eliminate false positives.

### Feature 3: Kinyarwanda Voice Output
Converts recognized signs into natural Kinyarwanda speech using either Google Text-to-Speech (online) or pyttsx3 (offline). Users can build complete sentences from individual signs and speak them aloud.

### Feature 4: Modern Dark-Themed GUI
A professional PyQt5 desktop application with real-time camera feed, recognition results panel, confidence meter, sentence builder, and a signing-state sidebar (READY / INTERPRETING indicators).

### Feature 5: Extensible Training Pipeline
Built-in data collection GUI and model training scripts allow easy expansion of the vocabulary. New signs can be added by recording samples via webcam and retraining the model — no programming knowledge required from end users.

---

## 5. Value Proposition

### What Makes HandsToVoice Unique?

1. **First-of-its-kind for Kinyarwanda**: No existing tool recognizes KSL and outputs Kinyarwanda speech. HandsToVoice is the first purpose-built solution for Rwanda's deaf community.

2. **Affordable and accessible**: Requires only a standard laptop/PC with a webcam — no specialized hardware, gloves, or sensors needed. Can run offline without internet.

3. **Locally relevant vocabulary**: The sign vocabulary is built around real Rwandan communication needs — greetings (Muraho, Amakuru), essential words (Amazi/Water, Ubufasha/Help, Muganga/Doctor), and culturally appropriate expressions.

4. **Real-time performance**: Sub-100ms inference time enables natural, conversational speed. The user signs, and the system speaks within a fraction of a second.

5. **Open and extensible**: Modular architecture allows new signs, languages, and features to be added without rebuilding the system. The training pipeline is included so communities can expand the vocabulary themselves.

---

## 6. Business Model

### Revenue Streams

| Stream | Description | Target |
|--------|-------------|--------|
| **Institutional Licensing** | Annual license for schools, hospitals, and government offices | Schools for the deaf, public hospitals, district offices |
| **NGO Partnerships** | Subsidized deployment through disability-focused NGOs | Handicap International, NUDOR, Rwanda Union of the Deaf |
| **Government Contracts** | Integration into national accessibility programs | MINALOC, Ministry of Health, Ministry of Education |
| **Training & Support** | Workshops for institutions adopting the system | Teachers, healthcare workers, social workers |
| **Freemium Mobile App** | Free basic version, premium features for advanced users | Individual deaf users and their families |

### Sustainability

- **Low operational cost**: Software-only solution with no hardware manufacturing.
- **Community-driven expansion**: Users and institutions can contribute training data to grow the vocabulary, reducing centralized development cost.
- **Grant eligibility**: Strong alignment with UN SDGs (Goal 4: Quality Education, Goal 10: Reduced Inequalities, Goal 3: Good Health) makes the project eligible for international development grants.

---

## 7. Social Impact

### Direct Impact

- **Empowering the deaf community**: Gives 72,000+ Rwandans with hearing/speech disabilities a voice — literally. They can now communicate with anyone, anywhere, using just their hands and a standard computer.
- **Inclusive education**: Students with hearing impairments can participate in mainstream classrooms without a dedicated interpreter. Teachers can understand their students' signed responses.
- **Healthcare access**: Patients can communicate symptoms, pain levels, and medical history to doctors and nurses, reducing misdiagnosis and improving treatment outcomes.

### Broader Impact

- **Raising awareness**: The visible use of the system normalizes sign language and increases public awareness of deaf culture and accessibility needs.
- **Job market access**: Deaf individuals can participate in job interviews and workplace communication, improving economic inclusion.
- **Model for Africa**: Rwanda becomes a leader in assistive technology, providing a replicable model for other African countries with similar challenges (each with their own sign language).
- **Family reconnection**: Family members who never learned sign language can now understand their deaf relatives, strengthening family bonds.

### Alignment with National Goals

- **Rwanda Vision 2050**: Technology-driven inclusive development.
- **National Policy for Persons with Disabilities**: Promotes equal access to services and opportunities.
- **ICT Sector Strategic Plan**: Innovation for social impact.

---

## 8. MVP (Minimum Viable Product)

### What We Built

The HandsToVoice MVP is a **fully functional desktop application** that demonstrates the complete sign-to-speech pipeline:

| MVP Component | Status | Description |
|--------------|--------|-------------|
| Hand Detection | ✅ Complete | MediaPipe-based 21-landmark detection at 30 FPS |
| Sign Classification | ✅ Complete | Neural network recognizing **12 KSL signs** |
| Voice Output | ✅ Complete | Kinyarwanda TTS (online + offline modes) |
| GUI Application | ✅ Complete | Dark-themed PyQt5 interface with live camera feed |
| Training Pipeline | ✅ Complete | Data collection GUI + model training scripts |
| Sentence Builder | ✅ Complete | Users can build and speak multi-sign sentences |

### Current Vocabulary (12 Signs)

| # | Sign | Kinyarwanda | English | Category |
|---|------|-------------|---------|----------|
| 1 | muraho | Muraho | Hello | Greetings |
| 2 | amakuru | Amakuru | How are you? | Greetings |
| 3 | yego | Yego | Yes | Common |
| 4 | oya | Oya | No | Common |
| 5 | murakoze | Murakoze | Thank you | Greetings |
| 6 | mbabarira | Mbabarira | Sorry / Excuse me | Common |
| 7 | ndagukunda | Ndagukunda | I love you | Common |
| 8 | amazi | Amazi | Water | Emergency |
| 9 | ubufasha | Ubufasha | Help | Emergency |
| 10 | muganga | Muganga | Doctor | Emergency |
| 11 | letter_c | C | Letter C | Alphabet |
| 12 | amaso | Amaso | Eyes | Body |

### What We Will Demo

1. **Live sign recognition**: The presenter performs KSL signs in front of the webcam, and the system identifies them in real time on screen.
2. **Voice output**: The system speaks the recognized Kinyarwanda word aloud (e.g., the signer shows "Muraho" and the system says "Muraho").
3. **Sentence building**: The presenter signs multiple words in sequence, building a sentence displayed on screen, then presses "Speak" to voice the complete sentence.
4. **Training pipeline**: We demonstrate how a new sign can be added to the system by collecting samples and retraining the model — showing the system's extensibility.

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     HandsToVoice System                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐│
│  │  Camera   │  │ MediaPipe│  │ TensorFlow│  │   PyQt5 GUI  ││
│  │  Capture  │─▶│ Hand     │─▶│ Classifier│─▶│  • Camera    ││
│  │  (OpenCV) │  │ Detector │  │  (Keras)  │  │  • Results   ││
│  └──────────┘  └──────────┘  └──────────┘  │  • Sentence  ││
│                                             │  • Controls  ││
│  ┌──────────────────────────────────────┐  │  • Sidebar   ││
│  │          Vocabulary Manager           │  └──────────────┘│
│  │  labels.json → Kinyarwanda mapping   │                    │
│  └──────────────────────────────────────┘                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │              Text-to-Speech Engine                        ││
│  │   gTTS (online) / pyttsx3 (offline) → Speaker Output     ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

*Prepared by:*
**NTAWUKURIRYAYO Frederic** (26-BK-RE-076) & **NYIRABACUMBITSI Melanie Celine** (26-BK-RE-055)

*University of Rwanda — College of Science and Technology*
*Department of Information and Communication Technology*
*April 2026*
