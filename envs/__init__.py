from gym.envs.registration import register

register(
    id='IntrinsicHandBlock-v0',
    entry_point='envs.handblock:IntrinsicHandBlockEnv',
    max_episode_steps=300
)
register(
    id='MagellanAnt-v2',
    entry_point='envs.ant:MagellanAntEnv',
    max_episode_steps=300
)

register(
    id='HyperAcrobot-v1',
    entry_point='envs.acrobat:AcrobotSparseContinuousEnv',
    max_episode_steps=500
)


register(
    id='MagellanHalfCheetah-v2',
    entry_point='envs.half_cheetah:MagellanHalfCheetahEnv',
    max_episode_steps=100
)

register(
    id='MagellanSparseMountainCar-v0',
    entry_point='envs.mountain_car:MagellanSparseContinuousMountainCarEnv',
    max_episode_steps=500
)
