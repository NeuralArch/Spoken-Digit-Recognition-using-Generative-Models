# Spoken Digit Recognition using Generative Models

Implemented GMM and HMM from scratch to recognize spoken digits (0–9) from MFCC features, achieving **91.6% test accuracy with HMM** and **80% with GMM**.

---

## Overview

This project builds a spoken digit recognition system using two probabilistic generative models — Gaussian Mixture Models (GMM) and Hidden Markov Models (HMM). Both models are implemented from scratch in Python without using any ML libraries like scikit-learn or hmmlearn.

The dataset consists of 3000 audio files (digits 0–9, spoken by 6 speakers). Models are trained on Speakers 1–4, validated on Speaker 5, and tested on the unseen Speaker 6.

---

## Models

### Gaussian Mixture Model (GMM)
- Treats each digit as a bag of feature vectors (no temporal modeling)
- 8-component diagonal GMM per digit
- Trained using the EM (Expectation-Maximization) algorithm
- Features: MFCC + delta + delta-delta coefficients

### Hidden Markov Model (HMM)
- Discrete left-to-right HMM with 7 states per digit
- MFCC features quantized into 64 symbols using K-Means clustering
- Trained using the Baum-Welch algorithm
- Decoded using the Forward algorithm

---

## Results

| Metric | HMM (Discrete) | GMM |
|---|---|---|
| Validation — Speaker 5 | 77.6% | 97.0% |
| Test — Speaker 6 | **91.6%** | **80.0%** |

HMM generalizes significantly better to an unseen speaker, as it captures the temporal structure of speech rather than just the acoustic features.

---

## Project Structure

```
.
├── GMM.py                      # GMM implementation and training
├── HMM.py                      # HMM implementation and training
├── spoken_digits_features.csv  # Extracted MFCC features
├── GMM_CM.png                  # GMM confusion matrix
├── GMM_Convergence.png         # GMM EM convergence plots
├── HMM_CM.png                  # HMM confusion matrix
├── HMM_Convergence.png         # HMM Baum-Welch convergence plots
└── Report.pdf                  # Full project report
```

---

## Features Used

- **MFCC** (Mel-Frequency Cepstral Coefficients)
- **Delta** coefficients (first-order derivatives)
- **Delta-Delta** coefficients (second-order derivatives)

---

## Key Observations

- **GMM** achieves high validation accuracy (97%) but drops on the unseen test speaker (80%), showing sensitivity to speaker-specific characteristics.
- **HMM** shows the opposite trend — lower validation (77.6%) but strong test accuracy (91.6%) — demonstrating better generalization across speakers.
- Both models converge smoothly within 20–30 iterations, confirming stable implementations.
- The most notable confusion in HMM is between digits **6 and 8**, which share similar phonetic endings.

---

## Requirements

```
numpy
pandas
matplotlib
```

---

## Usage

```bash
# Run GMM
python GMM.py

# Run HMM
python HMM.py
```

Both scripts read `spoken_digits_features.csv` and output confusion matrix and convergence plots.

---

## Course

Artificial Intelligence — April 2026

**Author:** Ayush Kanojiya (DA24B037)