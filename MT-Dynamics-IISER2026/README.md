# MT-Dynamics

*A modular, robust pipeline for precision microtubule tracing, dynamics analysis, and state modeling from fluorescence microscopy.*

## Overview

MT-Dynamics is an advanced image processing and machine learning system tailored for the analysis of microtubule growth and shrinkage (dynamic instability). The system solves the persistent challenge of accurately tracking low-contrast filamentous structures in noisy fluorescence videos by leveraging a dual-tier approach: robust classical algorithms (Frangi filtering, Otsu thresholding) combined with a fine-tuned U-Net architecture. Through this pipeline, researchers can ingest raw video data, extract precise semantic segmentations of microtubules, and compute essential biological metrics such as growth velocities and catastrophe frequencies.

## Technologies Used

* **Core Logic:** Python
* **Image Processing:** OpenCV, scikit-image, numpy, scipy
* **Machine Learning:** PyTorch, torchvision, segmentation-models-pytorch 
* **Data Handling:** pandas, datasets
* **Visualization:** matplotlib, Streamlit

## Purpose

The primary objective of this repository is to automate the analysis of microtubule dynamics, transitioning from manual annotation to high-throughput programmatic tracking. By accurately isolating the microtubule signal from background noise, this system enables precise modeling of dynamic transitions (growth, shrinkage, and pausing) critical for understanding cellular organization.

## Image Processing Approach

Our pipeline offers flexibility via two robust pathways:
1. **Classical Pipeline:** Utilizes Frangi vesselness filtering enhanced by CLAHE (Contrast Limited Adaptive Histogram Equalization) and morphology algorithms to extract binary skeletonized representations of microtubules.
2. **Deep Learning Pipeline:** Implements a U-Net model, fine-tuned on dynamically generated pseudo-labels from the classical pipeline, providing superior robustness against varying noise profiles and imaging artifacts.

## Quick Start
To process a video and extract dynamic metrics:
```bash
# 1. Enhance raw frames
python src/enhance_real_frames.py

# 2. Extract skeletons using the classical pipeline
python src/segment_real_classical.py

# 3. Model dynamics and extract metrics
python src/advanced_dynamics.py
```

## Directory Structure

```text
MT-Dynamics/
├── data/          # Raw images, enhanced frames, and processed skeletons
├── docs/          # Comprehensive system documentation & architectural diagrams
├── src/           # Core processing, ML training, tracking, and dynamics logic
├── dashboard/     # Interactive Streamlit application
├── results/       # Extracted metrics, ensemble statistics, and model weights
└── notebooks/     # Exploratory analysis and prototyping
```

## Training & Testing (U-Net) 

To reproduce the model weights using pseudo-labels:
```bash
# Generate pseudo-labels using classical methods
python src/generate_pseudo_labels.py

# Fine-tune the U-Net architecture
python src/finetune_unet_real.py

# Evaluate the model on a validation dataset
python src/evaluate_unet.py
```

## Future Enhancements
- Integration of a transformer-based tracker for handling extreme occlusion scenarios.
- Deployment of the Streamlit dashboard via high-availability cloud hosting structure.
- Implementation of real-time processing APIs for live microscopy feedback.

## Applications
- Accelerated pharmaceutical drug screening targeting microtubule stability.
- Educational visualization of cell division mechanisms.
- Automated quality control for synthetic filament synthesis.
