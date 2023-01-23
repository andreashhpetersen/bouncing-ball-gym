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
