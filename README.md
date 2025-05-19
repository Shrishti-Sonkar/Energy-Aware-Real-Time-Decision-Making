# Energy-Aware Real-Time Decision Making

This repository contains two Jupyter notebooks that demonstrate how to build, train, evaluate, and deploy energy-aware deep learning models for real-time decision making. Both notebooks instrument CodeCarbon to track emissions and energy usage.

- **`Untitled12.ipynb`**  
  Implements a ResNet-18–based pipeline:
  - Loads a pre-defined dataset (e.g., CIFAR-10 or ImageNet subset)
  - Defines and trains a ResNet-18 classifier with an “ERP” decision-making hook
  - Tracks energy consumption and CO₂ emissions during training
  - Evaluates the model on a test split
  - Demonstrates a real-time inference loop with dynamic energy-based decisions

- **`Untitled13.ipynb`**  
  Follows the same workflow but swaps in MobileNetV2 for a more lightweight architecture:
  - Defines `get_mobilenetv2_model()` in place of ResNet-18
  - Uses identical training and evaluation functions for apples-to-apples comparison
  - Tracks and reports energy/emissions metrics
  - Runs a real-time decision-maker that can adapt based on current power conditions

---

## Contents

- `Untitled12.ipynb` – ResNet-18 example  
- `Untitled13.ipynb` – MobileNetV2 example  
- `README.md` – this file  

---

## Prerequisites

- Python >= 3.8  
- [PyTorch](https://pytorch.org/) (with matching CUDA support, if available)  
- `torchvision`  
- `numpy`  
- `matplotlib`  
- [`codecarbon`](https://github.com/mlco2/codecarbon)  

Install dependencies via pip:

```bash
pip install torch torchvision numpy matplotlib codecarbon
