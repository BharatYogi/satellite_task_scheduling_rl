# Satellite Task Scheduling using Reinforcement Learning

## Overview
This project presents a simulation-based satellite task scheduling system developed as part of a Final Year Project. The system addresses the problem of selecting and executing satellite observation tasks under operational constraints such as limited battery, storage capacity, and time windows.

The project compares traditional heuristic scheduling methods with a reinforcement learning approach using a Deep Q-Network (DQN).


## Problem Statement
Satellites are required to perform multiple observation tasks, but due to limited onboard resources and strict time constraints, it is not possible to execute all tasks.

The challenge is to determine:
- Which tasks should be selected
- In what order they should be executed
- While maximising overall mission value and respecting constraints


## Objectives
- Simulate a realistic satellite scheduling environment  
- Implement multiple scheduling algorithms  
- Compare heuristic approaches with reinforcement learning  
- Evaluate performance using quantitative metrics  


## System Components

### 1. Task Generator
Generates a synthetic dataset of satellite tasks with attributes such as:
- Time window (start and end)
- Duration
- Priority
- Energy consumption
- Storage requirement
- Reward value


### 2. Simulation Environment
Models satellite constraints including:
- Limited battery capacity
- Limited storage capacity
- Task feasibility based on time windows
- Resource consumption over time


### 3. Scheduling Algorithms

#### Heuristic Methods
- First-Come-First-Served (FCFS)
- Highest Priority First
- Earliest Deadline First (EDF)
- Greedy Score-based Scheduler

#### Reinforcement Learning
- Deep Q-Network (DQN)
- Learns task selection strategy through interaction with the environment


### 4. Evaluation System
The system evaluates each scheduling approach using:
- Task completion ratio
- Total priority reward
- Constraint violations
- Battery utilisation
- Storage utilisation


## Results
The system generates the following outputs:

- `reward_curve.png` – training performance of DQN  
- `scheduler_comparison_reward.png` – reward comparison across methods  
- `completion_rate_comparison.png` – task completion comparison  
- `sample_schedule_timeline.png` – example scheduling timeline  
- `synthetic_tasks.csv` – generated dataset  


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

Run the following command:

```bash
python satellite_fyp_prototype.py

## **Author**
Bharat Nath Yogi  
BSc (Hons) Computer Science (Artificial Intelligence)  
University of Greenwich  
