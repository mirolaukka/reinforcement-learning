# 1. Import dependencies
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# 2. Load enviroment

visualize = False


environment_name = "CartPole-v0"

if visualize:
    env = gym.make(environment_name, render_mode="human")
else:
    env = gym.make(environment_name)

# Test Enviroment
# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info, _ = env.step(action)
#         print(n_state, reward, done, info)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()

# 3. Train an RL Model

# log_path = os.path.join('Training', 'Logs')

env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=9_000)

# 4. Save and Reload Model & Evaluate

# PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model_cartpole')

# model.save(PPO_path)

# del model

env = gym.make(environment_name, render_mode="human")


# model = PPO.load(PPO_path, env=env)


# 5. Evaluation

evaluated = evaluate_policy(
    model, env, n_eval_episodes=15, render=True, return_episode_rewards=True, warn=False)
print(evaluated)

for i, e in enumerate(evaluated[0]):
    print(i, e)

print("Mean:", round(sum(evaluated[0])/len(evaluated[0]), 2))
env.close()


# 6. Test Model

# obs = env.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info, _ = env.step(action)
#     env.render()
#     if done:
#         print('info', info)
#         break
# env.close()
