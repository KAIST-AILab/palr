from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.walker2d import Walker2dEnv

import numpy as np

class POAntEnv(AntEnv):
    def __init__(self, **kwargs):
        super(POAntEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POAntEnv, self).step(a)
        new_ob = ob[:13]
        info['full-state'] = ob
        
        return (
            new_ob, reward, done, info
        )

    def reset_model(self):
        ob = super(POAntEnv, self).reset_model()
        return ob[:13]

    def viewer_setup(self):
        super(POAntEnv, self).viewer_setup()


class POHopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        super(POHopperEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POHopperEnv, self).step(a)
        new_ob = ob[:5]
        info['full-state'] = ob
        
        return (
            new_ob, reward, done, info
        )
        
    def reset_model(self):
        ob = super(POHopperEnv, self).reset_model()
        return ob[:5]

    def viewer_setup(self):
        super(POHopperEnv, self).viewer_setup()


class POHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,8,9,10,11,12]
        super(POHalfCheetahEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POHalfCheetahEnv, self).step(a)
        new_ob = ob[self.poidx]
        info['full-state'] = ob
        
        return (new_ob, reward, done, info)

    def reset_model(self):
        ob = super(POHalfCheetahEnv, self).reset_model()
        return ob[self.poidx]

    def viewer_setup(self):
        super(POHalfCheetahEnv, self).viewer_setup()


class POWalker2dEnv(Walker2dEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,4,5,6,7]
        super(POWalker2dEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POWalker2dEnv, self).step(a)
        new_ob = ob[self.poidx]
        info['full-state'] = ob
        
        return (
            new_ob, reward, done, info
        )

    def reset_model(self):
        ob = super(POWalker2dEnv, self).reset_model()
        return ob[self.poidx]

    def viewer_setup(self):
        super(POWalker2dEnv, self).viewer_setup()
        