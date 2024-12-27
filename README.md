# Deep Learning Solutions for Doppler Radar Image Classification

This repository contains implementations of four different deep learning architectures for classifying Doppler radar images. The project compares Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), and Gated Recurrent Units (GRU) for their effectiveness in classifying targets from Doppler radar data.

## Significance

This research provides the first comprehensive comparative analysis of major deep learning architectures for Doppler radar target classification, addressing a critical gap in the field. The significance of this work lies in three key contributions:

1. **Novel Architectural Insights**: Demonstrates that recurrent architectures can achieve competitive performance in radar image classification, challenging conventional approaches

2. **Practical Design Guidelines**: Provides detailed architectural insights, design techniques and performance trade-offs between computational efficiency and classification accuracy, offering practical guidelines for resource-constrained applications

3. **Validated Methodology**: The study's rigorous evaluation methodology validates these approaches for practical deployment in security, surveillance, and autonomous system applications

## Dataset

The project uses an open-source dataset containing 17,485 Doppler radar images across three classes:
- Cars (5,720 samples)
- Drones (5,065 samples)
- People (6,700 samples)

Each sample consists of an 11 x 61 matrix, where:
- Columns represent Doppler frequencies
- Rows represent distance cells
- Values are expressed in decibel-milliwatts (dBm)

## Repository Structure

The repository contains separate implementation files for each architecture:
├── CNN/
│   ├── cnn_dev.ipynb       # Development and training notebook
│   ├── cnn_test31.ipynb    # Testing with random state 31
│   ├── cnn_test62.ipynb    # Testing with random state 62
│   └── cnn_test93.ipynb    # Testing with random state 93
├── RNN/
│   ├── rnn_dev.ipynb
│   ├── rnn_test31.ipynb
│   ├── rnn_test62.ipynb
│   └── rnn_test93.ipynb
├── LSTM/
│   ├── lstm_dev.ipynb
│   ├── lstm_test31.ipynb
│   ├── lstm_test62.ipynb
│   └── lstm_test93.ipynb
└── GRU/
├── gru_dev.ipynb
├── gru_test31.ipynb
├── gru_test62.ipynb
└── gru_test93.ipynb


## Performance Results

| Architecture | Avg Processing Time (ms/step) | Avg Accuracy |
|--------------|------------------------------|--------------|
| CNN          | 11.0                        | 96.0%        |
| RNN          | 7.0                         | 93.7%        |
| LSTM         | 25.7                        | 94.0%        |
| GRU          | 7.3                         | 93.7%        |

## Implementation Details

### CNN Architecture
- Two convolutional blocks with batch normalization and dropout
- Progressive filter expansion (32 → 64)
- Dense layers with gradual reduction (64 → 64 → 32 → 3)
- Optimized for balance between efficiency and accuracy

### RNN Architecture
- Dual SimpleRNN structure with 64 units per layer
- Tanh activation function
- Global average pooling for temporal feature aggregation
- Dense layers with 16 units each

### LSTM Architecture
- Dual LSTM blocks with 256 and 128 units
- Sigmoid activation functions
- Batch normalization and dropout for regularization
- Dense layers with ReLU activation

### GRU Architecture
- Two GRU blocks (128 → 64 units)
- Sigmoid activation functions
- Global average pooling
- Dense layers with 16 units each

## Key Findings

1. All architectures achieved robust performance with weighted average F1-scores above 0.93
2. CNN architecture provided the best balance of accuracy and processing time
3. LSTM showed strong performance but required significantly more processing time
4. GRU and RNN demonstrated efficient processing speeds while maintaining competitive accuracy
5. All architectures performed particularly well in classifying human targets

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Usage

Each architecture's development notebook (`*_dev.ipynb`) contains the complete implementation, including:
- Data preprocessing
- Model architecture
- Training configuration
- Performance evaluation

The test notebooks (`*_test*.ipynb`) contain validation experiments with different random states to ensure model robustness.

## Citation

If you use this code in your research, please cite:

@article{Developing a Deep Learning Based Target Classification Approach for Doppler Radars,2024,Mehmet Ali Turhan
title={Developing a Deep Learning Based Target Classification Approach for Doppler Radars},
author={Turhan, Mehmet Ali},
institution={Swinburne University of Technology},
year={2024}
}
