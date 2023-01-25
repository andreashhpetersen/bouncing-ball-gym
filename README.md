# Bouncing Ball environment for Gymnasium (OpenAI Gym)

A very simple environment where an agent has to decide when to hit a bouncing
ball so as to keep it bouncing for as long as possible but using as few hits as
possible.

## Installation

```sh
pip install git+ssh://git@github.com/andreashhp/bouncing-ball-gym.git
```

## Usage

The state space only consists of two continuous variables: the position and the
velocity of the ball. At each step, the agent can choose either to hit (action
`1`) or not to hit (action `0`). If the position of the ball is less than 4, or
if the velocity is less than -4, then the hit will have no effect except
generating a negative reward of -1. Otherwise, a hit will change the velocity to
-4 (and also give a negative reward of -1). If the ball dies out, a negative
reward of -1000 is given and the episode is terminated.

When initializing the environment, you can provide a `ts_size` argument to
specify how much time should elapse during each step (default is 1 second).


```python
import gymnasium as gym
import bouncing_ball_gym

env = gym.make('bouncing_ball/BouncingBallEnv-v0', ts_size=0.3)
obs, info = env.reset()
```

## Use with stable-baselines3

The `bouncing_ball` environment is built on
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) which is a fork of
OpenAI Gym v0.26.0 it does not work out of the box with the regular version of
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3). To use SB3,
you need to install it in the following way:

```sh
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
```

Then to use, for example, PPO from `stable-baselines3` you just do the
following:

```python
import sys
import gymnasium as gym
sys.modules['gym'] = gym  # needed to point sb3 to the right gym package

import bouncing_ball
from stable_baselines3 import PPO
from stable_baselines3 import make_vec_env

ts_size = 0.3
num_seconds = 120

env = make_vec_env(
    'bouncing_ball/BouncingBall-v0',
    env_kwargs={ 'ts_size': ts_size, 'render_mode': 'human' }
)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(250000)

hits = 0
obs = env.reset()
for t in range(int(num_seconds / ts_size)):
    action, _states = model.predict(obs, deterministic=True)

    hits += action

    obs, reward, done, info = env.step(action)

    if done:
        print('apparently, the model is not able to keep the ball bouncing!')
        break

print(hits)  # should be around 39
env.render()  # note: if you call render() and done == True, you will get an
              # error as sb3 resets the environment automatically
```

