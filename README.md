# Deep Learning-Based Neonatal Seizure Prediction and Detection

A system designed to automatically predict and detect neonatal seizures from multi-channel EEG recordings using a hybrid deep learning architecture. This project is developed for the Biomedical Signal Processing (BSP) course.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Our Solution](#our-solution)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Project Team](#project-team)

---

## Problem Statement

Neonatal seizures are a common neurological emergency affecting 1-5 per 1000 live births. Early and accurate detection is critical for preventing long-term brain damage. However, current methods based on manual EEG interpretation by neurologists are time-consuming, subjective, and not always available 24/7. This project aims to overcome the significant class imbalance (87.3% Normal vs. 12.7% Seizure) in EEG data to build a reliable automated detection system.

## Our Solution

We propose an intelligent automated system that leverages a hybrid **Convolutional Neural Network (CNN) + Long Short-Term Memory (LSTM)** architecture.

- The **CNN** extracts complex spatial features from the multi-channel EEG signals.
- The **LSTM** models the temporal evolution and patterns characteristic of a seizure.

This model is trained on a dataset augmented with novel, synthetically generated seizure data to overcome class imbalance and improve detection robustness.

## Key Features

1.  **Hybrid CNN+LSTM Architecture**: Optimized for extracting spatio-temporal features unique to neonatal brain signals.
2.  **Advanced Data Augmentation**: A novel pipeline using a **Finite Element Method (FEM)**-based approach to generate realistic synthetic seizure data. This involves:
    - Decomposing real seizures into physiological frequency bands (Delta, Theta, Alpha, Beta, Gamma).
    - Synthesizing new signals for each band.
    - Intelligently recombining the bands to create diverse and realistic seizure samples.
3.  **Real-Time Potential**: The architecture is designed to be efficient for real-time analysis in a clinical setting like the NICU.

## Technology Stack

- **Language**: `Python 3.x`
- **Core Libraries**:
    - `TensorFlow` / `Keras` (or `PyTorch`) for building the deep learning model.
    - `MNE-Python` for EEG data loading, preprocessing, and visualization.
    - `Scikit-learn` for baseline models and evaluation metrics.
    - `NumPy` & `Pandas` for data manipulation.
    - `Matplotlib` & `Seaborn` for plotting.

## Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

- Python 3.8 or higher

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rag-795/Neonatal-Seizure-Prediction-EEG.git
    cd Neonatal-Seizure-Prediction-EEG
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Team
- [Raghav N](https://github.com/Rag-795)
- [Hari Heman V K](https://github.com/HXMAN76)
- [Mathivanan S](https://github.com/Rag-795)
- [Rashwanth Ram](https://github.com/Rag-795)
