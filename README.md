# Snake RL

This project implements a Reinforcement Learning (RL) agent to play the classic Snake game using PyTorch. The agent learns to play Snake autonomously by interacting with the game environment and improving its performance over time.

## Features
- **Deep Q-Learning**: The agent uses a neural network to approximate Q-values and make decisions.
- **Custom Game Environment**: The Snake game is implemented using Pygame, allowing for easy modification and visualization.
- **Training Visualization**: Real-time plotting of scores and mean scores using Matplotlib.
- **Model Saving**: The best-performing model is saved automatically during training.

## File Structure
- `agent.py`: Contains the RL agent logic, including state representation, memory management, action selection, and training loop.
- `game.py`: Implements the Snake game environment, including game mechanics, rendering, and collision detection.
- `model.py`: Defines the neural network architecture (`Linear_QNet`) and the Q-learning trainer (`QTrainer`).
- `helper.py`: Provides utility functions for plotting training progress.

## How It Works
1. **State Representation**: The agent observes the game state, including dangers, direction, and food location.
2. **Action Selection**: Actions are chosen using an epsilon-greedy strategy to balance exploration and exploitation.
3. **Training**: The agent trains using both short-term and long-term memory, applying the Bellman Equation to update Q-values.
4. **Game Loop**: The agent interacts with the game, receives rewards, and updates its policy based on experience.
5. **Visualization**: Training progress is visualized in real-time.

## Requirements
- Python 3.7+
- PyTorch
- Pygame
- Matplotlib
- NumPy

Install dependencies with:
```bash
pip install torch pygame matplotlib numpy
```

## Running the Project
To start training the RL agent, run:
```bash
python agent.py
```
This will launch the training loop and display a live plot of the agent's performance.

## Customization
- **Game Parameters**: Modify `BLOCK_SIZE`, `SPEED`, or window size in `game.py`.
- **Model Architecture**: Change the neural network layers in `model.py`.
- **Training Parameters**: Adjust `MAX_MEMORY`, `BATCH_SIZE`, or learning rate in `agent.py`.
