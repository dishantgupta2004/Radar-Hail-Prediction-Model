# Radar Hail Prediction Model

## Overview
This repository contains an implementation of a hybrid transformer-physics-GenAI framework for physics-informed hailstorm prediction using deep neural operators. The model integrates transformer-based deep learning with physics-informed neural operators and generative modeling to create a physics-aware hailstorm prediction framework.

## Features
- Combines transformer encoders for spatiotemporal feature extraction
- Incorporates atmospheric physical laws through Fourier Neural Operators (FNO) and Deep Operator Networks (DeepONets)
- Models uncertainty through variational latent spaces
- Integrates core physical equations:
  - Advection-diffusion equation
  - Continuity equation
  - Helmholtz equation

## Repository Structure
```
RADAR-HAIL-PREDICTION-MODEL/
│
├── .vs/                            # Visual Studio configuration
├── Chaos Theory/                   # Chaos theory analysis and implementation
│   ├── Analysis/
│   │   ├── Implementation_and_Analysis.ipynb
│   │   └── multidimenstional_Lorentz_system.ipynb
│   └── neural_ode_parameter_variation.png
│
├── data/                           # Data storage and processing
│   ├── processed/                  # Processed radar data
│   ├── simulated/                  # Simulated data
│   └── complete_final_dataset_IMERG_Nov_05.h5  # Main dataset
│
├── dgmr/                           # Deep Generative Model of Radar
│   ├── layers/                     # DGMR layer implementations
│   │   ├── __init__.py
│   │   ├── Attention.py            # Attention mechanisms
│   │   ├── ConvGRU.py              # Convolutional GRU implementation
│   │   ├── CoordConv.py            # Coordinate convolution
│   │   └── utils.py                # Utility functions for layers
│   ├── __init__.py
│   ├── common.py                   # Common DGMR components
│   ├── dgmr.py                     # Main DGMR model implementation
│   ├── discriminators.py           # DGMR discriminator components
│   ├── generators.py               # DGMR generator components
│   ├── hub.py                      # Model hub integration
│   └── losses.py                   # Loss functions for DGMR
│
├── models/                         # Model implementations
│   └── myenv/                      # Python environment
│
├── notebooks/                      # Jupyter notebooks
│   └── analysis/                   # Analysis notebooks
│       ├── 01_Introduction_and_Background.ipynb
│       ├── 02_Data_Analysis_Visualization.ipynb
│       ├── 03_Implementation_Scattering.ipynb
│       └── 04_FEM.ipynb
│
├── src/                            # Source code
│   ├── data/                       # Data handling
│   │   ├── __init__.py
│   │   ├── dataloader.py           # Data loading utilities
│   │   └── radar_data.py           # Radar data processing
│   │
│   ├── integration/                # Integration components
│   │   ├── __init__.py
│   │   ├── hybrid_model.py         # Hybrid physics-ML model
│   │   └── pipeline.py             # Model pipeline
│   │
│   ├── pidl/                       # Physics-informed deep learning
│   │   ├── __init__.py
│   │   ├── inference.py            # Inference utilities
│   │   ├── model.py                # Main PIDL model implementation
│   │   ├── physics_loss.py         # Physics-based loss functions
│   │   └── train.py                # Training code
│   │
│   └── utils/                      # Utilities
│       ├── __init__.py
│       ├── io.py                   # Input/output utilities
│       ├── metrics.py              # Evaluation metrics
│       └── visualization.py        # Visualization tools
│
├── tests/                          # Unit tests
└── train/                          # Training scripts
```

## Installation

```bash
# Clone the repository

# Create and activate a virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

The model expects radar data in a specific format. You can process your raw data using the utilities in `src/data/`:

```python
from src.data.dataloader import prepare_radar_data

# Process raw radar data
prepare_radar_data(input_path="raw_data/", output_path="data/processed/")
```

### Training

To train the hybrid model:

```python
from src.pidl.train import train_model

# Train the model
train_model(
    data_path="data/complete_final_dataset_IMERG_Nov_05.h5",
    output_dir="models/trained/",
    epochs=100,
    batch_size=16,
    learning_rate=1e-4
)
```

### Inference

To run inference with a trained model:

```python
from src.pidl.inference import predict

# Run prediction
predictions = predict(
    model_path="models/trained/model.pt",
    input_data="data/test_data.h5"
)
```

## Model Architecture

The hybrid framework consists of five main components:

1. **Spatiotemporal Encoder**: Extracts relevant features from input meteorological data
2. **Physics-Aware Module**: Incorporates atmospheric physics using neural operators
3. **Latent Variable Module**: Captures uncertainty and hidden atmospheric structure
4. **Fusion Layer**: Combines physics-based and latent representations
5. **Decoder**: Generates the final hailstorm prediction maps

## Physics Components

The model incorporates three key physical processes relevant to hailstorm formation:

1. **Advection-diffusion**: Represents the transport and diffusion of temperature, humidity, and other scalar quantities
   ```
   ∂ϕ/∂t + u·∇ϕ = κ∇²ϕ
   ```

2. **Continuity equation**: Ensures conservation of mass in fluid flow
   ```
   ∂ρ/∂t + ∇·(ρu) = 0
   ```

3. **Helmholtz equation**: Models wave propagation in the atmosphere
   ```
   ∇²ψ + k²ψ = 0
   ```

## Evaluation Metrics

The model is evaluated using several metrics:

- Root Mean Square Error (RMSE)
- Intersection over Union (IoU)
- Continuous Ranked Probability Score (CRPS)
- Structural Similarity Index (SSIM)
- Critical Success Index (CSI)

## Citation

If you use this code in your research, please cite our work:

```
@article{gupta2024hybrid,
  title={Hybrid Transformer-Physics-GenAI Framework for Physics-Informed Hailstorm Prediction Using Deep Neural Operators},
  author={Gupta, Dishant and Mokta, Ajay},
  journal={},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This work was supported by a grant from the National Science Foundation.