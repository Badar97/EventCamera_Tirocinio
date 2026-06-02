# Face Tracking with Event-Based Vision and DVS Cameras

This repository contains the code and documentation for my Bachelor's thesis project in Computer and Automation Engineering at Università Politecnica delle Marche (UNIVPM), academic year 2020/2021.

The project focuses on face tracking using event-based vision, combining asynchronous events and traditional frames from Dynamic Vision Sensors (DVS).

---

## Objectives

- Develop an algorithm to track human faces using data from event-based cameras.
- Combine event streams and frame data to improve accuracy and robustness.
- Explore facial mesh tracking for eyes, lips, and face landmarks from DVS input.
- Demonstrate real-time applicability in dynamic or low-light conditions.

---

## Main Features

- Hybrid processing of event-based and frame-based data
- Modular scripts for each face area:
  - Eyes (`eyeProcessing.py`)
  - Lips (`lipsProcessing.py`)
  - Face (`faceProcessing.py`)
  - Combined face mesh (`totProcessing.py`)
- Face landmark extraction and mesh visualization with MediaPipe and OpenCV
- Motion analysis through velocity and direction calculation on facial keypoints
- Reusable utility functions for JSON parsing, landmark selection, filtering, and plotting

---

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- DVS / event-camera data processing

---

## Project Structure

```zsh 
    EventCamera_Tirocinio/
    ├── json/                   # JSON configuration files
    ├── utilities/              # Utility scripts and helper functions
    ├── eyeMeshing.py           # Eye mesh processing
    ├── eyeProcessing.py        # Eye event processing
    ├── faceMeshing.py          # Face mesh processing
    ├── faceProcessing.py       # Face event processing
    ├── lipsMeshing.py          # Lips mesh processing
    ├── lipsProcessing.py       # Lips event processing
    ├── totMeshing.py           # Combined mesh processing
    ├── totProcessing.py        # Combined event processing
    ├── Tesi_Ali_Waqar_Badar.pdf# Thesis document
    └── README.md               # Project documentation
```

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Badar97/EventCamera_Tirocinio.git
   cd EventCamera_Tirocinio
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the main dependencies used by the scripts**:
   ```bash
   pip install numpy matplotlib opencv-python mediapipe
   ```

   Some scripts may also require the `dv` Python bindings from iniVation to read DVS/AEDAT data streams.

4. **Run one of the main scripts**, for example:
   ```bash
   python faceProcessing.py
   ```

> Note: The original thesis workflow depends on DVS recordings and JSON files produced during the experimental setup.

## Thesis
You can find the full thesis report (in Italian) here:
- Tesi_Ali_Waqar_Badar.pdf

It includes:

- Theoretical background on Event Cameras and DVS
- Algorithm explanation
- Implementation steps
- Results and conclusions

  
## Authors
- [Ali Waqar Badar](https://github.com/Badar97)

> This project was developed for academic research and educational purposes.
