# Implementation Code for PALR (Past Action Leakage Regularization) 

- This code contains implementation codes of PALR, which is used to induce experimental results in the paper (under review); please do not distribute.
- The code contains PALR and its baseline 4 methods (BC, FCA, MINE, RAP).

### 1. Prerequisites

- To run this code, first install the anaconda virtual environment:

```
conda env create -f environment.yaml
conda activate palr
```

- (optional) Download D4RL dataset:
```
python download_d4rl_dataset.py
```

### 2. Train & Evaluate PALR
- Train imitation policies using `main.py`.
For the ease of hyperparameter search, `pid` pass into main code and `pid`-th configuration of the overall grid will be executed.
Note that the default setting is:
```
    methodlist        = ['BC', 'RAP', 'FCA', 'MINE', 'PALR']
    envlist           = ['Hopper', 'Walker2d', 'HalfCheetah', 'Ant']
    stacksizelist     = [2, 4]
    seedlist          = [0, 1, 2, 3, 4]    
```

To execute 0-th configuration, i.e. `method='BC', env='Hopper', stacksize=2, seed=0`, run:
```
python train.py --pid=0
``` 

### 3. Notes
- Results are averaged over 5 different seed.
- Our code implementation is based on the following public repositories:
    - rlkit : https://github.com/rail-berkeley/rlkit
    - FCA : https://github.com/AlvinWen428/fighting-copycat-agents.git
    - MINE implementation : https://github.com/mohith-sakthivel/mine-pytorch.git
    - HSCIC implementation : https://github.com/namratadeka/circe.git
