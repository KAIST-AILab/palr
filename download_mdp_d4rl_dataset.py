import os
os.environ['D4RL_DATASET_DIR'] = './dataset'

import gym
import d4rl

def generate_mdp_dataset(envname):
    # mdpdata_filepath = 'dataset/mdps'
    print("--Start:", envname)
    env = gym.make(envname)
    dataset = env.get_dataset()
    print("--Done:", envname)

if __name__ == "__main__":
    envlist = ['hopper-expert', 'halfcheetah-expert', 'walker2d-expert', 'ant-expert']
    versionlist = ['-v2']  #'-v0', '-v1', 
    
    for env in envlist:
        for version in versionlist:
            envname = env + version
            generate_mdp_dataset(envname)
