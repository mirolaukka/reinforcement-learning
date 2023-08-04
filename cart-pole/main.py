import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Load environment
visualize = False
environment_name = "CartPole-v0"
if visualize:
    env = gym.make(environment_name, render_mode="human")
else:
    env = gym.make(environment_name)

# Train an RL Model
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=9_000)

# Evaluate the Model
evaluated = evaluate_policy(model, env, n_eval_episodes=15,
                            render=True, return_episode_rewards=True, warn=False)
print(evaluated)

# Print the evaluation results
for i, e in enumerate(evaluated[0]):
    print(i, e)
print("Mean:", round(sum(evaluated[0])/len(evaluated[0]), 2))
env.close()
