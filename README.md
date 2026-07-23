# Neural Network Error Mitigation for Quantum Communication

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Research](https://img.shields.io/badge/Research-Quantum%20Computing-purple.svg)

A research implementation demonstrating how lightweight neural networks can improve the performance of two fundamental quantum communication protocols under realistic noisy conditions.

This repository contains complete simulations of:

- Quantum Teleportation
- Superdense Coding

along with neural-network-based post-processing techniques for mitigating structured quantum noise.

---

## Overview

Quantum communication protocols perform perfectly in ideal environments but degrade significantly in the presence of hardware imperfections, gate errors, and measurement noise.

This project investigates whether a small classical neural network can learn recurring noise patterns and improve protocol performance without modifying the underlying quantum algorithm.

Unlike many previous studies, the neural network:

- does **not** replace any quantum computation
- does **not** know the noise parameters
- only receives realistic finite-shot measurement outcomes

The goal is to determine **when learned error mitigation actually helps**.

---

## Repository Structure

```
.
├── README.md
├── quantum_teleportation_nn_correction.ipynb
└── superdense_coding_nn_decoding.ipynb
```

### quantum_teleportation_nn_correction.ipynb

Implements

- Exact quantum teleportation simulator
- Bloch sphere representation
- Systematic gate miscalibration
- Finite-shot tomography
- Neural network correction
- Fidelity evaluation
- Control experiment using random rotations

---

### superdense_coding_nn_decoding.ipynb

Implements

- Exact superdense coding protocol
- Biased Pauli noise channel
- Asymmetric readout noise
- Finite-shot measurements
- Majority-vote decoder
- Neural network decoder
- Accuracy comparison across different shot counts

---

## Features

- Exact tensor-based quantum simulation
- No quantum computing framework required
- PyTorch neural networks
- Finite-shot measurement simulation
- Structured noise models
- Control experiments
- Reproducible training
- Publication-quality figures

---

## Methodology

### Quantum Teleportation

Pipeline:

```
Unknown Quantum State
        ↓
Teleportation Circuit
        ↓
Systematic Rotation Error
        ↓
Finite-shot Tomography
        ↓
Neural Network
        ↓
Corrected Bloch Vector
```

Evaluation metric:

- Mean State Fidelity

---

### Superdense Coding

Pipeline:

```
2-bit Classical Message
        ↓
Bell State Encoding
        ↓
Biased Pauli Noise
        ↓
Readout Error
        ↓
Finite-shot Measurements
        ↓
Neural Network Decoder
        ↓
Recovered Message
```

Evaluation metric:

- Classification Accuracy

---

## Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib torch notebook
```

or

```bash
pip install -r requirements.txt
```

---

## Running the Notebooks

Clone the repository

```bash
git clone https://github.com/yourusername/your-repository.git
```

Move into the repository

```bash
cd your-repository
```

Launch Jupyter

```bash
jupyter notebook
```

Run either notebook:

- `quantum_teleportation_nn_correction.ipynb`
- `superdense_coding_nn_decoding.ipynb`

---

## Results

### Quantum Teleportation

The neural network learns to compensate for systematic gate miscalibration, improving average teleportation fidelity from approximately

```
0.93 → 0.99
```

under structured noise.

When the noise is completely random, the network provides no improvement, confirming that the model only learns recurring error patterns.

---

### Superdense Coding

The learned decoder consistently outperforms majority-vote decoding in the low-shot regime.

Typical improvement:

| Shots | Majority Vote | Neural Network |
|--------|--------------:|---------------:|
| 5 | 82% | 87% |
| 10 | 91% | 94% |
| 20 | 96% | 98% |

As the number of shots increases, both approaches converge toward perfect decoding.

---

## Research Paper

This repository accompanies the research work:

> **Neural-Network Error Mitigation for Quantum Teleportation and Superdense Coding**

The work investigates when classical machine learning can effectively mitigate structured quantum noise without access to the underlying noise model.

---

## Future Work

Potential extensions include

- Multi-qubit teleportation
- Surface-code decoding
- Hardware calibration data
- IBM Quantum experiments
- Qiskit implementation
- Quantum error correction benchmarks
- Larger neural architectures
- Device-specific noise models

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{chaudhary2025nnquantum,
  title={Neural-Network Error Mitigation for Quantum Teleportation and Superdense Coding},
  author={Hritik Chaudhary and Gokul K.C.},
  year={2025}
}
```

---

## Author

**Hritik Chaudhary**

Department of Mathematics  
Kathmandu University  
Nepal

Email:
```
kitirhhritik@gmail.com
```

---

## Acknowledgements

This work was carried out at

**Department of Mathematics**

School of Science

Kathmandu University

Special thanks to the quantum computing and machine learning research communities for the foundational work that inspired this project.

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Hritik Chaudhary

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

⭐ If you found this project useful, consider starring the repository.
