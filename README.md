# Satellite Task Scheduling using Reinforcement Learning

## Overview
This project presents a simulation-based satellite task scheduling system developed as part of a Final Year Project. The system addresses the problem of selecting and executing satellite observation tasks under operational constraints such as limited battery, storage capacity, and time windows.

The project compares traditional heuristic scheduling methods with a reinforcement learning approach using a Deep Q-Network (DQN).


## Problem Statement
Satellites must perform multiple observation tasks, but due to limited onboard resources and strict time constraints, not all tasks can be executed.

The challenge is to determine:
- Which tasks should be selected  
- In what order they should be executed  
- While maximising mission value under constraints  


## Objectives
- Simulate a satellite scheduling environment  
- Implement multiple scheduling algorithms  
- Compare heuristic methods with reinforcement learning  
- Evaluate performance using defined metrics  


## System Components

### Task Generator
Generates a synthetic dataset with:
- Time windows  
- Priority levels  
- Energy consumption  
- Storage requirements  


### Simulation Environment
Models constraints such as:
- Battery limits  
- Storage limits  
- Task feasibility  


### Scheduling Algorithms

#### Heuristic Methods
- First-Come-First-Served (FCFS)  
- Highest Priority First  
- Earliest Deadline First (EDF)  
- Greedy Score-based Scheduler  

#### Reinforcement Learning
- Deep Q-Network (DQN)  


### Evaluation
Performance is measured using:
- Task completion ratio  
- Total priority reward  
- Constraint violations  
- Battery utilisation  
- Storage utilisation  


## Results
The system generates:

- reward_curve.png  
- scheduler_comparison_reward.png  
- completion_rate_comparison.png  
- sample_schedule_timeline.png  
- synthetic_tasks.csv  


## Key Findings
- Heuristic methods achieved stable and higher performance  
- The DQN agent produced feasible schedules but lower rewards  
- Reinforcement learning showed learning behaviour but requires further optimisation  


## Technologies Used
- Python  
- NumPy  
- Matplotlib  
- Reinforcement Learning (DQN)  


## How to Run

```bash
python satellite_fyp_prototype.py

## Author

Bharat Nath Yogi  
BSc (Hons) Computer Science (Artificial Intelligence)  
University of Greenwich  
Final Year Project (2026)
