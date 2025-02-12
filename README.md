# Exploration of Multi-Agent Reinforcement Learning (MARL)

## Overview
Multi-Agent Reinforcement Learning (MARL) has become a crucial area in artificial intelligence, particularly in environments where multiple autonomous agents interact and learn simultaneously. This project explores MARL techniques by implementing and evaluating three key reinforcement learning algorithms:
- **Proximal Policy Optimization (PPO)**
- **Multi-Agent PPO (MAPPO)**
- **Deep Q-Networks (DQN)**

Additionally, we introduce **MAPPO-CR**, a novel algorithm that integrates MAPPO with a recommendation system for optimizing experience replay selection.

## Features
- **Comparative analysis of MARL algorithms**
- **Unity ML-Agents framework for training**
- **Self-play mechanisms and Multi-Agent POsthumous Credit Assignment (MA-POCA)**
- **Optimized experience replay selection using a recommendation system**
- **Evaluation on a multi-agent soccer environment**

## Project Structure
```
├── src
│   ├── base_agent.py       # Base class for all agents
│   ├── ppo_agent.py        # PPO implementation
│   ├── mappo_agent.py      # MAPPO implementation
│   ├── dqn_agent.py        # DQN implementation
│   ├── mappo_cr_agent.py   # MAPPO-CR implementation
│
├── environment
│   ├── soccer_twos         # Unity ML-Agents soccer environment
│
├── results
│   ├── training_metrics.png
│   ├── comparison_metrics.png
│
├── README.md
├── requirements.txt
```

## Installation
### Requirements
- **Python 3.8+**
- **Unity ML-Agents (v2.0 or higher)**
- **PyTorch (v1.9 or higher)**
- **NumPy, Matplotlib, Pandas, Seaborn**

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/bsdivith/marl.git
   cd marl
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Unity ML-Agents:
   ```bash
   cd environment/soccer_twos
   ```
   Follow the Unity ML-Agents [installation guide](https://github.com/Unity-Technologies/ml-agents) to set up the environment.

## Training and Evaluation
### Run Training
Train the agents using:
```bash
python src/train.py --algorithm mappo_cr
```
Supported algorithms: `ppo`, `mappo`, `dqn`, `mappo_cr`

### View Training Results
Results are stored in the `results/` directory and can be visualized using:
```bash
python src/plot_results.py
```

## Results and Analysis
- MAPPO-CR achieved **faster convergence** and **higher stability** compared to standard MARL algorithms.
- Self-play mechanisms significantly improved adaptability and strategy learning.
- MAPPO-CR demonstrated superior coordination through optimized experience replay.

## Future Work
- **Scaling MAPPO-CR** to larger, more dynamic environments.
- **Integrating decentralized MARL techniques** for improved scalability.
- **Exploring meta-learning approaches** for adaptive policy selection.

## Authors
- **Divith B S** ([GitHub](https://github.com/bsdivith) | [LinkedIn](https://www.linkedin.com/in/divith-b-s))
- **Anirudh Sajith** ([GitHub](https://github.com/An1rud))
- **Harsh Manalel** ([GitHub](https://github.com/HarshManalel))
 
## Acknowledgments
This project was carried out under the supervision of **Dr. G Naveen Babu** at **Dayananda Sagar University**.



