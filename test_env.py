import envs
import gym


if __name__ == '__main__':
    env = gym.make('AccConcatHopper-v0')
    obs = env.reset()
    env.step(env.action_space.sample())
    print(obs.shape)
    env = gym.make('AccConcatAnt-v0')
    obs = env.reset()
    env.step(env.action_space.sample())
    print(obs.shape)
    env = gym.make('AccConcatWalker2d-v0')
    obs = env.reset()
    env.step(env.action_space.sample())
    print(obs.shape)
    env = gym.make('AccConcatHalfCheetah-v0')
    obs = env.reset()
    env.step(env.action_space.sample())
    print(obs.shape)
    