from argparse import ArgumentParser
import numpy as np
import gym
import h5py

def preprocess_pomdp_dataset(envname, stacksize):
    # mdpdata_filepath = 'dataset/mdps'
    print("--Start:", envname, stacksize)
    data_loadpath = f'dataset/pomdps/{envname}_expert.hdf5'
    data_savepath = f'dataset/pomdps/{envname}_expert_h{stacksize}.hdf5'

    mdpfile = h5py.File(data_loadpath, 'r')

    # keys : ['actions', 'observations', 'rewards', 'terminals', 'timeouts']
    done = False

    observations = np.array(mdpfile['observations'])
    terminals = np.array(mdpfile['terminals'])
    timeouts = np.array(mdpfile['timeouts'])
    rewards = np.array(mdpfile['rewards'])
    actions = np.array(mdpfile['actions'])
    mdpfile.close()

    pomdpfile = h5py.File(data_savepath, 'w')

    obs_dim = observations.shape[-1]
    n_data = observations.shape[0]
    new_observations_list = []
    
    idx_from_initial_state = 0
    num_trajs = 0

    for i in range(n_data):
        if idx_from_initial_state < stacksize:
            new_observation = np.zeros(obs_dim * stacksize)
            new_observation_ = np.concatenate(observations[i-idx_from_initial_state: i+1])
            new_observation[-(idx_from_initial_state+1) * obs_dim:] = new_observation_
        else:
            new_observation = np.concatenate(observations[i+1-stacksize:i+1])

        new_observations_list.append(new_observation)

        idx_from_initial_state += 1
        if terminals[i] or timeouts[i]:
            idx_from_initial_state = 0
            num_trajs += 1

    new_observations = np.array(new_observations_list)

    pomdpfile.create_dataset('observations', data=new_observations)
    pomdpfile.create_dataset('actions',   data=actions)
    pomdpfile.create_dataset('rewards',   data=rewards)
    pomdpfile.create_dataset('terminals', data=terminals)
    pomdpfile.create_dataset('timeouts',  data=timeouts)

    pomdpfile.close()
    
    print("--Done:", envname, stacksize)
    print("--num_trajs:", num_trajs)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--pid", help="process_id", default=0, type=int)
    # envlist = ['halfcheetah-expert-v0', 'hopper-expert-v0', 'walker2d-expert-v0', 'ant-expert-v0']
    
    # indx = list(np.arange(20))
    envlist = ['ant']       #'hopper', 'ant', 'walker2d', 'halfcheetah']
    stacksizelist = [5]     #, indx[:13], indx[:8], indx[:4] + indx[8:13]]

    for envname in envlist:
        for stacksize in stacksizelist:
            preprocess_pomdp_dataset(envname, stacksize)
