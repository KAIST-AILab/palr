from gym.envs.registration import register

# POMDP Environments
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

# POMDP Environments with stacked observations
register(
    id='POStackedAnt-v0',
    entry_point='envs.pomujoco:POStackedAntEnv',
)

register(
    id='POStackedHopper-v0',
    entry_point='envs.pomujoco:POStackedHopperEnv',
)

register(
    id='POStackedWalker2d-v0',
    entry_point='envs.pomujoco:POStackedWalker2dEnv',
)

register(
    id='POStackedHalfCheetah-v0',
    entry_point='envs.pomujoco:POStackedHalfCheetahEnv',
)

# POMDP Environments with action augmented in observation
register(
    id='POAAnt-v0',
    entry_point='envs.pomujoco:POAAntEnv',
)

register(
    id='POAHopper-v0',
    entry_point='envs.pomujoco:POAHopperEnv',
)

register(
    id='POAWalker2d-v0',
    entry_point='envs.pomujoco:POAWalker2dEnv',
)

register(
    id='POAHalfCheetah-v0',
    entry_point='envs.pomujoco:POAHalfCheetahEnv',
)

# POMDP Environments with action augmented in observation and stacked observations
register(
    id='POAStackedAnt-v0',
    entry_point='envs.pomujoco:POAStackedAntEnv',
)

register(
    id='POAStackedHopper-v0',
    entry_point='envs.pomujoco:POAStackedHopperEnv',
)

register(
    id='POAStackedWalker2d-v0',
    entry_point='envs.pomujoco:POAStackedWalker2dEnv',
)

register(
    id='POAStackedHalfCheetah-v0',
    entry_point='envs.pomujoco:POAStackedHalfCheetahEnv',
)

# MDP Environments with stacked observations
register(
    id='StackedAnt-v0',
    entry_point='envs.pomujoco:StackedAntEnv',
)

register(
    id='StackedHopper-v0',
    entry_point='envs.pomujoco:StackedHopperEnv',
)

register(
    id='StackedWalker2d-v0',
    entry_point='envs.pomujoco:StackedWalker2dEnv',
)

register(
    id='StackedHalfCheetah-v0',
    entry_point='envs.pomujoco:StackedHalfCheetahEnv',
)

# MDP Environments with stacked observations with Actions
register(
    id='AStackedAnt-v0',
    entry_point='envs.pomujoco:AStackedAntEnv',
)

register(
    id='AStackedHopper-v0',
    entry_point='envs.pomujoco:AStackedHopperEnv',
)

register(
    id='AStackedWalker2d-v0',
    entry_point='envs.pomujoco:AStackedWalker2dEnv',
)

register(
    id='AStackedHalfCheetah-v0',
    entry_point='envs.pomujoco:AStackedHalfCheetahEnv',
)

# MDP Environments with concatenated with Acceleration (action diff.)
register(
    id='AccConcatAnt-v0',
    entry_point='envs.pomujoco:AccConcatAntEnv',
)

register(
    id='AccConcatHopper-v0',
    entry_point='envs.pomujoco:AccConcatHopperEnv',
)

register(
    id='AccConcatWalker2d-v0',
    entry_point='envs.pomujoco:AccConcatWalker2dEnv',
)

register(
    id='AccConcatHalfCheetah-v0',
    entry_point='envs.pomujoco:AccConcatHalfCheetahEnv',
)
