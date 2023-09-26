import math
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# gravity
G = -9.81


class BouncingBallEnv(gym.Env):
    metadata = { 'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None, ts_size=0.3, max_n_steps=400):

        self.observation_space = gym.spaces.Box(
            low=np.array([0, -25], dtype=np.float32),
            high=np.array([50, 25], dtype=np.float32),
        )
        self.action_space = gym.spaces.Discrete(2)

        self.ts = ts_size
        self.max_n_steps = max_n_steps
        self.render_mode = render_mode

    def _get_obs(self):
        return np.array([self.p, self.v], dtype=np.float32)

    def _get_info(self):
        return { 'time': self.time }

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        self.time = 0.0
        self.steps_taken = 0

        self.p = 7 + self.np_random.uniform(0, 3)
        self.v = 0.0

        # for rendering
        self.positions = []
        self.velocities = []
        self.actions = []

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):

        terminated = False
        self.steps_taken += 1
        self.time += self.ts

        # if an action is taken and position is at least 4
        if action == 1 and self.p >= 4.0:

            # hit when ball is going up
            if self.v >= 0.0:
                flip = -(0.9 + self.np_random.uniform(0, 0.1))
                self.v = (flip * self.v) - 4.0

            # hit when ball is already falling
            elif self.v >= -4.0:
                self.v = -4.0

        # new candidate state
        new_v = self.v + (self.ts * G)
        new_p = self.p + (self.v * self.ts) + 0.5 * (G * np.square(self.ts))

        # ball bounces!
        if new_p <= 0.0 and self.v < 0.0:

            # solve for t when p == 0
            D = np.sqrt(np.square(self.v) - (2 * G * self.p))
            t = max((-self.v + D) / G, (-self.v - D) / G)

            # velocity when the ball hits the ground
            new_v = self.v + (t * G)

            # flip velocity at bounce and loose some momentum
            new_v *= -(0.85 + self.np_random.uniform(0, 0.12))
            if new_v <= 1:
                terminated = True

            # new position (starting from 0 as we have just bounced)
            new_p = new_v * (self.ts - t) + 0.5 * (G * np.square(self.ts - t))

            # new velocity
            new_v += (self.ts - t) * G

        self.p = new_p
        self.v = new_v

        self.positions.append(self.p)
        self.velocities.append(self.v)
        self.actions.append(action)

        obs = self._get_obs()
        info = self._get_info()
        # terminated = self.p <= 0
        truncated = self.steps_taken >= self.max_n_steps
        reward = -1 * action - 1000 * terminated

        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        N = len(self.positions)

        fig, axs = plt.subplots(nrows=2, figsize=(5,3))
        p_ax, v_ax = axs[0], axs[1]

        min_pos, max_pos = -1, max(self.positions)
        min_vel, max_vel = min(self.velocities), max(self.positions)

        p_ax.set_title('Position')
        p_ax.set_xlim(0, N)
        p_ax.set_ylim(min_pos, max_pos)

        v_ax.set_title('Velocity')
        v_ax.set_xlim(0, N)
        v_ax.set_ylim(min_vel, max_vel)

        x = np.arange(N)

        line_pos = p_ax.plot([0], self.positions[:1], color='b', lw=1)[0]
        line_vel = v_ax.plot([0], self.velocities[:1], color='r', lw=1)[0]

        def animate(i):
            line_pos.set_xdata(x[:i+1])
            line_pos.set_ydata(self.positions[:i+1])

            line_vel.set_xdata(x[:i+1])
            line_vel.set_ydata(self.velocities[:i+1])

            # actions
            if self.actions[i] == 1:
                p_ax.scatter(i-1, self.positions[i-1], color='g', marker='x')
                v_ax.scatter(i-1, self.positions[i-1], color='g', marker='x')

                p_ax.plot([i-1, i-1], [min_pos, max_pos], color='grey')
                v_ax.plot([i-1, i-1], [min_vel, max_vel], color='grey')

        plt.grid(True)
        animation_1 = animation.FuncAnimation(
            fig, animate, frames=N, interval=1, repeat=False
        )
        plt.show()
        plt.close()
