import json

import os
import argparse
import wandb

import logging
import logging.config
import random
import numpy as np
import torch

logger = logging.getLogger("__main__")
def set_logging(config_file, level='INFO'):
    level = getattr(logging, level)
    with open(config_file, 'r') as json_file:
        log_config = json.load(json_file)
        logging.config.dictConfig(log_config)
    logger.setLevel(logging.DEBUG)
    
def make_workspace(config, output_dir='', fmt='', delimiter='-'):
    if fmt == '':
        n = getattr(config, output_dir) + '/' + 'debug'
        os.makedirs(n, exist_ok=True)

    else:
        name = []
        for f in fmt.split(delimiter):
            if hasattr(config, f):
                name += [str(getattr(config, f))]
            else:
                name += [f]
        name = delimiter.join(name)

        name = getattr(config, output_dir) + '/' + name 
        i = 0
        while os.path.exists(n := name + delimiter + str(i)):
            i += 1 

        os.makedirs(n)
    setattr(config, output_dir, n)

def init_wandb(config, name='test', project_name='exp', debug=False):
    """
    :return wandb_logger: Wandb_Run 
    >>> wandb_logger = init_wandb(config, name=os.path.basename(config.output_dir), project_name='Transformer')
    >>> wandb_logger(dict{...})
    """
    wandb_logger = wandb.init(name=name ,config=config, entity="sent-word", resume="allow",
                                project=project_name)
         
    return wandb_logger

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('NN', add_help=False)
    parser.add_argument('--output_dir', default="results", type=str, help="hihihi") 
    parser.add_argument('--lr', default="0.3", type=float, help="hihihi") 
    parser.add_argument('--model', default="transforming", type=str, help="hihihi") 
    args = parser.parse_args()
    print(args)
    init_wandb(args, name='hi', project_name='test')
    # make_workspace(args, "output_dir", delimiter='')
    print(args)