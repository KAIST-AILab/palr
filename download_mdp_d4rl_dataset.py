from argparse import ArgumentParser
import numpy as np
import gym
import d4rl
import os
os.environ['D4RL_DATASET_DIR'] = '/tmp'


def generate_mdp_dataset(envname):
    # mdpdata_filepath = 'dataset/mdps'
    print("--Start:", envname)
    env = gym.make(envname)
    dataset = env.get_dataset()
    print("--Done:", envname)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--pid", help="process_id", default=0, type=int)
    envlist = ['hopper-expert', 'halfcheetah-expert', 'walker2d-expert', 'ant-expert']
    versionlist = ['-v2']  #'-v0', '-v1', 
    
    for env in envlist:
        for version in versionlist:
            envname = env + version
            generate_mdp_dataset(envname)
