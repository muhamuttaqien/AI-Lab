from gym.envs.registration import register

register(
    id='MyEnv-v0',
    entry_point='my_env.envs:CustomEnv',
    max_episode_steps=2000,
)