# CartPole Reinforcement Learning

This repository contains a Python script that demonstrates training a reinforcement learning (RL) model using the Stable Baselines3 library and OpenAI Gym's CartPole environment.

## Getting Started

To get started, make sure you have Python installed on your system. Additionally, you need to install the required dependencies: gymnasium, stable_baselines3, and any other dependencies they may require.

```bash
pip install gymnasium stable_baselines3
```

## Running the Code

To run the script, execute the following command in your terminal or command prompt:

```bash
python main.py
```

## Code Explanation

The script performs the following steps:

1. **Import Dependencies**: The required libraries are imported, including gymnasium for the environment, and Stable Baselines3 for RL algorithms.

2. **Load Environment**: The CartPole-v0 environment from OpenAI Gym is loaded. You can set the `visualize` variable to `True` if you want to see the environment while running, or `False` to run it without visualization.

3. **Train an RL Model**: The script uses Proximal Policy Optimization (PPO) algorithm with the Multi-Layer Perceptron (MLP) Policy for training the RL model. The training is performed for 9,000 time steps.

4. **Evaluate the Model**: After training, the model is evaluated for its performance using the `evaluate_policy` function. The evaluation is done for 15 episodes. The evaluation results, including episode rewards, are printed.

5. **Print Evaluation Results**: The script then prints the individual episode rewards and calculates the mean reward for all episodes.
