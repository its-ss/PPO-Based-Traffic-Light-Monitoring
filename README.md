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
