# PPO-Based Traffic Light Monitoring System

## Overview

This project implements a **Proximal Policy Optimization (PPO)-based Reinforcement Learning (RL) model** to optimize traffic light control in mixed traffic environments. The model is designed to minimize waiting times and queue lengths while improving traffic throughput using the **SUMO** traffic simulator.

## Project Structure

The repository contains the following key files:
- **`PPO.ipynb`** - Implements the PPO reinforcement learning model to optimize traffic light control at a single intersection.
- **`Fixed.ipynb`** - Implements a fixed traffic light monitoring system to compare performance with the RL-based model.

## Requirements

To run the notebooks, install the following dependencies:

```bash
pip install jupyter numpy matplotlib gym tensorflow torch stable-baselines3 sumo
```

## SUMO Installation

SUMO (Simulation of Urban Mobility) is required to run the simulations. Install it using:

Linux:
```bash
sudo apt-get install sumo
```

macOS:
```bash
brew install sumo
```

Windows:
Download and install SUMO from [Eclipse SUMO](https://sumo.dlr.de/docs/Downloads.php).

## How to Run

Open Jupyter Notebook:
```bash
jupyter notebook
```

1. Open `Fixed.ipynb` to analyze the fixed traffic light monitoring system.
2. Open `PPO.ipynb` to train and evaluate the PPO-based model.
3. Run all the cells in sequence.

## Methodology

### Environment Setup:
SUMO simulates a single intersection with mixed traffic conditions.

### Reinforcement Learning (PPO):
- **State Space:** Number of vehicles, waiting time, and current traffic signal phase.
- **Action Space:** Choosing an optimal green light phase.
- **Reward Function:** Minimizes waiting time and queue length.

### Comparison:
The performance of fixed-time vs. RL-based adaptive control is evaluated.

## Authors

- Abdul Jumail (MIT Manipal)
- Suyash Shukla (MIT Manipal)
- Rashmi Laxmikant Malghan (MIT Manipal)

## License

🚨 This project is private and copyrighted. Unauthorized distribution or reproduction is prohibited.

## Contact

For any inquiries, please contact:
📧 rashmi.malghan@manipal.edu
