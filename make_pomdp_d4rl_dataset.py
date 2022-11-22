from argparse import ArgumentParser
import numpy as np
import gym
import h5py

def generate_pomdp_dataset(envname, indx):
    # mdpdata_filepath = 'dataset/mdps'
    print("--Start:", envname, indx)
    mdpdata_loadpath = f'dataset/mdps/{envname}_expert.hdf5'
    pomdpdata_savepath = f'dataset/pomdps/{envname}_expert.hdf5'

    mdpfile = h5py.File(mdpdata_loadpath, 'r')
    pomdpfile = h5py.File(pomdpdata_savepath, 'w')

    # keys : ['actions', 'observations', 'rewards', 'terminals', 'timeouts']
    new_observations = mdpfile['observations'][:, indx]
    pomdpfile.create_dataset('observations', data=new_observations)
    pomdpfile.create_dataset('actions',   data=mdpfile['actions'])
    pomdpfile.create_dataset('rewards',   data=mdpfile['rewards'])
    pomdpfile.create_dataset('terminals', data=mdpfile['terminals'])
    pomdpfile.create_dataset('timeouts',  data=mdpfile['timeouts'])

    pomdpfile.close()
    mdpfile.close()
    print("--Done:", envname, indx)


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--pid", help="process_id", default=0, type=int)
    # envlist = ['halfcheetah-expert-v0', 'hopper-expert-v0', 'walker2d-expert-v0', 'ant-expert-v0']
    
    indx = list(np.arange(20))
    envlist = ['hopper']#, 'ant', 'walker2d', 'halfcheetah']
    idxlist = [indx[:5]]#, indx[:13], indx[:8], indx[:4] + indx[8:13]]

    for envname, indx in zip(envlist, idxlist):
        generate_pomdp_dataset(envname, indx)
