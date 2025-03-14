# Whisper Flask App

## Description
A Flask-based application that uses Whisper (ONNX) for voice recognition. This app enables real-time voice-to-text transcription.

## Hardware Requirements
* **Platform**: QCS6490
* **CPU**: Octa-Core Kryo 670
* **GPU**: Qualcomm Adreno 643

## Software Requirements
* **Operating System**: Ubuntu 20.04 (arm64)
* **SNPE SDK Version**: v2.26.0.240828

## Setup Steps

### 1. Switch to Admin Mode
Ensure you have superuser access before proceeding with the installation:
```bash
su
oelinux
```

### 2. Clone and Install Whisper App Project
```
apt install git
https://github.com/Kaiwei0323/whisper-test.git
cd whisper-test
```

### 3. Environment Setup
```
chmod +x install.sh
./install.sh
```

### 4. Export XDG_RUNTIME_DIR environment variable
```
export XDG_RUNTIME_DIR=/run/user/0
```

### 5. Run Application
```
python3.10 app.py
```
