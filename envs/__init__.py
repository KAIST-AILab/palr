from gym.envs.registration import register
# ConfoundedContinuousMountainCar-v0 : augmented random states 

register(
    id='POAnt-v0',
    entry_point='envs.pomujoco:POAntEnv',
)

register(
    id='POHopper-v0',
    entry_point='envs.pomujoco:POHopperEnv',
)

register(
    id='POWalker2d-v0',
    entry_point='envs.pomujoco:POWalker2dEnv',
)

register(
    id='POHalfCheetah-v0',
    entry_point='envs.pomujoco:POHalfCheetahEnv',
)
