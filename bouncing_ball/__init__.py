from gymnasium.envs.registration import register

register(
    id='bouncing_ball/BouncingBall-v0',
    entry_point='bouncing_ball.envs:BouncingBallEnv',
    max_episode_steps=None,
)
