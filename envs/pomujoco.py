from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.walker2d import Walker2dEnv

import numpy as np

'''
Partially Observable Environment 
'''
class POAntEnv(AntEnv):
    def __init__(self, **kwargs):
        self.init_true_state = None
        super(POAntEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(POAntEnv, self).step(a)
        new_ob = ob[:13]

        info['next_state'] = ob
        
        return (
            new_ob, reward, done, info
        )

    def reset_model(self):
        self.init_true_state = super(POAntEnv, self).reset_model()
        ob = self.init_true_state [:13]
        return ob

    def viewer_setup(self):
        super(POAntEnv, self).viewer_setup()


class POHopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        self.init_true_state = None
        super(POHopperEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(POHopperEnv, self).step(a)
        new_ob = ob[:5]

        info['next_state'] = ob        
        
        return (
            new_ob, reward, done, info
        )
        
    def reset_model(self):
        self.init_true_state = super(POHopperEnv, self).reset_model()
        ob = self.init_true_state[:5]
        return ob

    def viewer_setup(self):
        super(POHopperEnv, self).viewer_setup()


class POHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,8,9,10,11,12]
        self.init_true_state = None
        super(POHalfCheetahEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POHalfCheetahEnv, self).step(a)
        new_ob = ob[self.poidx]
        info['next_state'] = ob
        
        return (new_ob, reward, done, info)

    def reset_model(self):
        self.init_true_state = super(POHalfCheetahEnv, self).reset_model()
        ob = self.init_true_state[self.poidx]
        return ob

    def viewer_setup(self):
        super(POHalfCheetahEnv, self).viewer_setup()


class POWalker2dEnv(Walker2dEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,4,5,6,7]
        self.init_true_state = None
        super(POWalker2dEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POWalker2dEnv, self).step(a)
        new_ob = ob[self.poidx]
        info['next_state'] = ob
        
        return (
            new_ob, reward, done, info
        )

    def reset_model(self):
        self.init_true_state = super(POWalker2dEnv, self).reset_model()
        ob = self.init_true_state[self.poidx]
        return ob

    def viewer_setup(self):
        super(POWalker2dEnv, self).viewer_setup()



'''
Partially Observable Environment with Stacked Frames
'''
class POStackedAntEnv(AntEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None
        super(POStackedAntEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POStackedAntEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs()[:13])
        new_ob = np.concatenate((self.prev_obs, ob[:13]))

        info['next_state'] = ob
        self.prev_obs = ob[:13]
        
        return (
            new_ob, reward, done, info
        )

    def reset_model(self):
        ob = super(POStackedAntEnv, self).reset_model()[:13]
        self.prev_obs = np.zeros_like(ob)        
        stacked_ob = np.concatenate((self.prev_obs, ob))

        return stacked_ob

    def viewer_setup(self):
        super(POStackedAntEnv, self).viewer_setup()


class POStackedHopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None
        super(POStackedHopperEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POStackedHopperEnv, self).step(a)
        
        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs()[:5])
        
        new_ob = np.concatenate((self.prev_obs, ob[:5]))
        
        info['next_state'] = ob
        self.prev_obs = ob[:5]
        
        return (
            new_ob, reward, done, info
        )
        
    def reset_model(self):
        ob = super(POStackedHopperEnv, self).reset_model()[:5]
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate((self.prev_obs, ob))

        return stacked_ob

    def viewer_setup(self):
        super(POStackedHopperEnv, self).viewer_setup()


class POStackedHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,8,9,10,11,12]
        self.prev_obs = None
        super(POStackedHalfCheetahEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POStackedHalfCheetahEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs()[self.poidx])

        new_ob = np.concatenate((self.prev_obs, ob[self.poidx]))
        info['next_state'] = ob
        self.prev_obs = ob[self.poidx]
        
        return (new_ob, reward, done, info)

    def reset_model(self):
        ob = super(POStackedHalfCheetahEnv, self).reset_model()[self.poidx]
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate((self.prev_obs, ob))

        return stacked_ob

    def viewer_setup(self):
        super(POStackedHalfCheetahEnv, self).viewer_setup()


class POStackedWalker2dEnv(Walker2dEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,4,5,6,7]
        self.prev_obs = None
        super(POStackedWalker2dEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POStackedWalker2dEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs()[self.poidx])
        
        new_ob = np.concatenate((self.prev_obs, ob[self.poidx]))
        info['next_state'] = ob
        self.prev_obs = ob[self.poidx]
        
        return (new_ob, reward, done, info)

    def reset_model(self):
        ob = super(POStackedWalker2dEnv, self).reset_model()[self.poidx]
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate((self.prev_obs, ob))

        return stacked_ob

    def viewer_setup(self):
        super(POStackedWalker2dEnv, self).viewer_setup()


'''
Partially Observable Environment with Previous Actions
'''
class POAAntEnv(AntEnv):
    def __init__(self, **kwargs):
        super(POAAntEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POAAntEnv, self).step(a)
        new_ob = np.concatenate( (ob[:13], a) )
        info['next_state'] = ob
        info['next_observation'] = ob[:13]
        
        return (
            new_ob, reward, done, info
        )

    def reset_model(self):
        ob = super(POAAntEnv, self).reset_model()
        return np.concatenate( (ob[:13], np.zeros(self.action_space.shape[0])) ) 

    def viewer_setup(self):
        super(POAAntEnv, self).viewer_setup()


class POAHopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        super(POAHopperEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POAHopperEnv, self).step(a)
        new_ob = np.concatenate( (ob[:5], a) )
        info['next_state'] = ob
        info['next_observation'] = ob[:5]
        
        return (
            new_ob, reward, done, info
        )
        
    def reset_model(self):
        ob = super(POAHopperEnv, self).reset_model()
        return np.concatenate( (ob[:5], np.zeros(self.action_space.shape[0])) )

    def viewer_setup(self):
        super(POAHopperEnv, self).viewer_setup()


class POAHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,8,9,10,11,12]
        super(POAHalfCheetahEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POAHalfCheetahEnv, self).step(a)
        new_ob = np.concatenate((ob[self.poidx], a))
        info['next_state'] = ob
        info['next_observation'] = ob[self.poidx]
        
        return (new_ob, reward, done, info)

    def reset_model(self):
        ob = super(POAHalfCheetahEnv, self).reset_model()
        return np.concatenate( (ob[self.poidx], np.zeros(self.action_space.shape[0])) )

    def viewer_setup(self):
        super(POAHalfCheetahEnv, self).viewer_setup()


class POAWalker2dEnv(Walker2dEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,4,5,6,7]
        super(POAWalker2dEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POAWalker2dEnv, self).step(a)
        new_ob = np.concatenate((ob[self.poidx], a))
        info['next_state'] = ob
        info['next_observation'] = ob[self.poidx]
        
        return (
            new_ob, reward, done, info
        )

    def reset_model(self):
        ob = super(POAWalker2dEnv, self).reset_model()
        return np.concatenate( (ob[self.poidx], np.zeros(self.action_space.shape[0])) )

    def viewer_setup(self):
        super(POAWalker2dEnv, self).viewer_setup()
        

'''
Partially Observable Environment with Previous Actions and Stacked Observation
'''
class POAStackedAntEnv(AntEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None #np.zeros_like(self.observation_space.sample()[:13])
        super(POAStackedAntEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(POAStackedAntEnv, self).step(a)
        # new_ob = np.concatenate( (ob[:13], a) )

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs()[:13])

        new_ob = np.concatenate((self.prev_obs, a, ob[:13]))

        info['next_state'] = ob
        info['next_observation'] = ob[:13]
        self.prev_obs = ob[:13]
        
        return (new_ob, reward, done, info)

    def reset_model(self):
        ob = super(POAStackedAntEnv, self).reset_model()[:13]
        init_a = np.zeros(self.action_space.shape[0])
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate((self.prev_obs, init_a, ob))

        return stacked_ob  # np.concatenate( (ob[:13], np.zeros(self.action_space.shape[0])) ) 

    def viewer_setup(self):
        super(POAStackedAntEnv, self).viewer_setup()


class POAStackedHopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        # self.prev_obs = np.zeros_like(self.observation_space.sample()[:5])
        self.prev_obs = None
        super(POAStackedHopperEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(POAStackedHopperEnv, self).step(a)
        # new_ob = np.concatenate( (ob[:5], a) )
        
        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs()[:5])
        new_ob = np.concatenate((self.prev_obs, a, ob[:5]))

        info['next_state'] = ob
        info['next_observation'] = ob[:5]
        self.prev_obs = ob[:5]
        
        return (new_ob, reward, done, info)
        
    def reset_model(self):
        ob = super(POAStackedHopperEnv, self).reset_model()[:5]
        init_a = np.zeros(self.action_space.shape[0])
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate( (self.prev_obs, init_a, ob) )
        return stacked_ob

    def viewer_setup(self):
        super(POAStackedHopperEnv, self).viewer_setup()


class POAStackedHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,8,9,10,11,12]
        self.prev_obs = None #np.zeros_like(self.observation_space.sample()[self.poidx])
        super(POAStackedHalfCheetahEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(POAStackedHalfCheetahEnv, self).step(a)
        # new_ob = np.concatenate((ob[self.poidx], a))

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs()[self.poidx])
        new_ob = np.concatenate((self.prev_obs, a, ob[self.poidx]))

        info['next_state'] = ob
        info['next_observation'] = ob[self.poidx]
        self.prev_obs = ob[self.poidx]
        
        return (new_ob, reward, done, info)

    def reset_model(self):
        ob = super(POAStackedHalfCheetahEnv, self).reset_model()[self.poidx]
        init_a = np.zeros(self.action_space.shape[0])
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate( (self.prev_obs, init_a, ob) )
        return stacked_ob

    def viewer_setup(self):
        super(POAStackedHalfCheetahEnv, self).viewer_setup()


class POAStackedWalker2dEnv(Walker2dEnv):
    def __init__(self, **kwargs):
        self.poidx = [0,1,2,3,4,5,6,7]
        self.prev_obs = None # np.zeros_like(self.observation_space.sample()[self.poidx])
        super(POAStackedWalker2dEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(POAStackedWalker2dEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs()[self.poidx])
        
        # new_ob = np.concatenate((ob[self.poidx], a))
        new_ob = np.concatenate((self.prev_obs, a, ob[self.poidx]))
        
        info['next_state'] = ob
        info['next_observation'] = ob[self.poidx]
        
        return (new_ob, reward, done, info)

    def reset_model(self):
        ob = super(POAStackedWalker2dEnv, self).reset_model()[self.poidx]
        init_a = np.zeros(self.action_space.shape[0])
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate( (self.prev_obs, init_a, ob) )
        return stacked_ob

    def viewer_setup(self):
        super(POAStackedWalker2dEnv, self).viewer_setup()
        


'''
Stacked State Environment 
'''
class StackedAntEnv(AntEnv):
    def __init__(self, **kwargs):        
        self.prev_obs = None
        super(StackedAntEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(StackedAntEnv, self).step(a)        

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        info['current_state'] = self.prev_obs
        info['next_state'] = ob

        stacked_ob = np.concatenate((self.prev_obs, ob))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )

    def reset_model(self):
        ob = super(StackedAntEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate((self.prev_obs, ob))

        return stacked_ob

    def viewer_setup(self):
        super(StackedAntEnv, self).viewer_setup()


class StackedHopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None
        super(StackedHopperEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(StackedHopperEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        info['current_state'] = self.prev_obs
        info['next_state'] = ob        
        
        stacked_ob = np.concatenate((self.prev_obs, ob))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
        
    def reset_model(self):
        ob = super(StackedHopperEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate((self.prev_obs, ob))

        return stacked_ob

    def viewer_setup(self):
        super(StackedHopperEnv, self).viewer_setup()


class StackedHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None
        super(StackedHalfCheetahEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(StackedHalfCheetahEnv, self).step(a)
        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        info['current_state'] = self.prev_obs
        info['next_state'] = ob        
        
        stacked_ob = np.concatenate((self.prev_obs, ob))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
    
    def reset_model(self):
        ob = super(StackedHalfCheetahEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate((self.prev_obs, ob))

        return stacked_ob

    def viewer_setup(self):
        super(StackedHalfCheetahEnv, self).viewer_setup()


class StackedWalker2dEnv(Walker2dEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None        
        super(StackedWalker2dEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(StackedWalker2dEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        info['current_state'] = self.prev_obs
        info['next_state'] = ob        
        
        stacked_ob = np.concatenate((self.prev_obs, ob))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
    
    def reset_model(self):        
        ob = super(StackedWalker2dEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        stacked_ob = np.concatenate((self.prev_obs, ob))

        return stacked_ob

    def viewer_setup(self):
        super(StackedWalker2dEnv, self).viewer_setup()

'''
Stacked State Environment with Actions
'''
class AStackedAntEnv(AntEnv):
    def __init__(self, **kwargs):        
        self.prev_obs = None
        super(AStackedAntEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(AStackedAntEnv, self).step(a)        

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        info['current_state'] = self.prev_obs
        info['next_state'] = ob

        stacked_ob = np.concatenate((self.prev_obs, a, ob))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )

    def reset_model(self):
        ob = super(AStackedAntEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        init_a = np.zeros(self.action_space.shape[0])
        stacked_ob = np.concatenate((self.prev_obs, init_a, ob))

        return stacked_ob

    def viewer_setup(self):
        super(AStackedAntEnv, self).viewer_setup()


class AStackedHopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None
        super(AStackedHopperEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(AStackedHopperEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        info['current_state'] = self.prev_obs
        info['next_state'] = ob        
        
        stacked_ob = np.concatenate((self.prev_obs, a, ob))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
        
    def reset_model(self):
        ob = super(AStackedHopperEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        init_a = np.zeros(self.action_space.shape[0])
        stacked_ob = np.concatenate((self.prev_obs, init_a, ob))

        return stacked_ob

    def viewer_setup(self):
        super(AStackedHopperEnv, self).viewer_setup()


class AStackedHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None
        super(AStackedHalfCheetahEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(AStackedHalfCheetahEnv, self).step(a)
        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        info['current_state'] = self.prev_obs
        info['next_state'] = ob        
        
        stacked_ob = np.concatenate((self.prev_obs, a, ob))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
    
    def reset_model(self):
        ob = super(AStackedHalfCheetahEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        init_a = np.zeros(self.action_space.shape[0])
        stacked_ob = np.concatenate((self.prev_obs, init_a, ob))

        return stacked_ob

    def viewer_setup(self):
        super(AStackedHalfCheetahEnv, self).viewer_setup()


class AStackedWalker2dEnv(Walker2dEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None        
        super(AStackedWalker2dEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(AStackedWalker2dEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        info['current_state'] = self.prev_obs
        info['next_state'] = ob        
        
        stacked_ob = np.concatenate((self.prev_obs, a, ob))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
    
    def reset_model(self):        
        ob = super(AStackedWalker2dEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        init_a = np.zeros(self.action_space.shape[0])
        stacked_ob = np.concatenate((self.prev_obs, init_a, ob))

        return stacked_ob

    def viewer_setup(self):
        super(AStackedWalker2dEnv, self).viewer_setup()


'''
Accel. concatenated Environment with Actions
'''
class AccConcatAntEnv(AntEnv):
    def __init__(self, **kwargs):        
        self.prev_obs = None
        self.vel_idx = slice(19,27) # 19 ~ 26 : 8
        super(AccConcatAntEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(AccConcatAntEnv, self).step(a)        

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        acc = ob[self.vel_idx] - self.prev_obs[self.vel_idx]

        info['current_state'] = self.prev_obs
        info['next_state'] = ob
        info['acceleration'] = acc

        stacked_ob = np.concatenate((ob, acc))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )

    def reset_model(self):
        ob = super(AccConcatAntEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        acc = ob[self.vel_idx] - self.prev_obs[self.vel_idx]

        stacked_ob = np.concatenate((ob, acc))

        return stacked_ob

    def viewer_setup(self):
        super(AccConcatAntEnv, self).viewer_setup()


class AccConcatHopperEnv(HopperEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None
        self.vel_idx = slice(7,11) # 7 ~ 10 : 4dim
        super(AccConcatHopperEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(AccConcatHopperEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        acc = ob[self.vel_idx] - self.prev_obs[self.vel_idx]

        info['current_state'] = self.prev_obs
        info['next_state'] = ob
        info['acceleration'] = acc
        
        stacked_ob = np.concatenate((ob, acc))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
        
    def reset_model(self):
        ob = super(AccConcatHopperEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)        
        acc = ob[self.vel_idx] - self.prev_obs[self.vel_idx]
        stacked_ob = np.concatenate((ob, acc))

        return stacked_ob

    def viewer_setup(self):
        super(AccConcatHopperEnv, self).viewer_setup()


class AccConcatHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None
        self.vel_idx = list(np.arange(0,4)) + list(np.arange(8,13)) # 0~3, 8~12 : 4+5=9
        super(AccConcatHalfCheetahEnv, self).__init__(**kwargs)

    def step(self, a):
        ob, reward, done, info = super(AccConcatHalfCheetahEnv, self).step(a)
        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        acc = ob[self.vel_idx] - self.prev_obs[self.vel_idx]
        info['current_state'] = self.prev_obs
        info['next_state'] = ob
        info['acceleration'] = acc
        
        stacked_ob = np.concatenate((ob, acc))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
    
    def reset_model(self):
        ob = super(AccConcatHalfCheetahEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        acc = ob[self.vel_idx] - self.prev_obs[self.vel_idx]
        stacked_ob = np.concatenate((ob, acc))

        return stacked_ob

    def viewer_setup(self):
        super(AccConcatHalfCheetahEnv, self).viewer_setup()


class AccConcatWalker2dEnv(Walker2dEnv):
    def __init__(self, **kwargs):
        self.prev_obs = None        
        self.vel_idx = slice(8, 17) # 8~16 : 9
        super(AccConcatWalker2dEnv, self).__init__(**kwargs)        

    def step(self, a):
        ob, reward, done, info = super(AccConcatWalker2dEnv, self).step(a)

        if self.prev_obs is None:
            self.prev_obs = np.zeros_like(self._get_obs())

        acc = ob[self.vel_idx] - self.prev_obs[self.vel_idx]
        info['current_state'] = self.prev_obs
        info['next_state'] = ob
        info['acceleration'] = acc
        
        stacked_ob = np.concatenate((ob, acc))
        self.prev_obs = ob
        
        return (
            stacked_ob, reward, done, info
        )
    
    def reset_model(self):        
        ob = super(AccConcatWalker2dEnv, self).reset_model()
        self.prev_obs = np.zeros_like(ob)
        acc = ob[self.vel_idx] - self.prev_obs[self.vel_idx]
        stacked_ob = np.concatenate((ob, acc))

        return stacked_ob

    def viewer_setup(self):
        super(AccConcatWalker2dEnv, self).viewer_setup()
