from argparse import ArgumentParser
import numpy as np
import gym
import h5py
import d4rl
import os


def generate_mdp_dataset(envname):
    # mdpdata_filepath = 'dataset/mdps'
    print("--Start:", envname)
    env = gym.make(envname)
    dataset = env.get_dataset()
    
    keys = dataset.keys()
    
    os.makedirs('logs', exist_ok=True)
    # f1 = open("logs/dataset_info1.txt", "a")
    # f2 = open("logs/dataset_info2.txt", "a")
    f3 = open("logs/dataset_keys.txt", "a")
    
    # calculate episode returns
    f3.write(f'---- {envname} ----\n')
    f3.write(str(keys) + '\n')
    # n_dataset = dataset['rewards'].shape[0]
    
    # rewards = dataset['rewards']
    # terminals = dataset['terminals']
    # timeouts = dataset['timeouts']
    
    # nonzeroidx_list = list(np.nonzero(np.logical_or(timeouts, terminals))[0])
    # start_idx = 0
    # ret_list = []
    
    # for nonzeroidx in nonzeroidx_list:
    #     end_idx = nonzeroidx
        
    #     ret = np.sum(rewards[start_idx:end_idx+1])
    #     ret_list.append(ret)
        
    #     start_idx = end_idx+1        

    # f1.write(f'---- {envname} ----\n')
    # f1.write(str(ret_list) + '\n')
    
    # mean = np.mean(ret_list)
    # stderr = np.std(ret_list) / np.sqrt(len(ret_list))
    # f2.write(f'{envname} : {mean} +- {stderr}\n')
    
    # print((f'{envname} : {mean} +- {stderr}\n'))
    
    # f1.close()
    # f2.close()
    f3.close()
    
    print("--Done:", envname)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--pid", help="process_id", default=0, type=int)
    envlist = ['halfcheetah-expert', 'hopper-expert', 'walker2d-expert', 'ant-expert']
    versionlist = ['-v0', '-v1', '-v2']
    
    indx = list(np.arange(20))
    # envlist = ['hopper']#, 'ant', 'walker2d', 'halfcheetah']
    # idxlist = [indx[:5]]#, indx[:13], indx[:8], indx[:4] + indx[8:13]]

    for env in envlist:
        for version in versionlist:
            envname = env + version
            generate_mdp_dataset(envname)
