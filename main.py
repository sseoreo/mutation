from tkinter.ttk import Label
import torch
import torch.nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import argparse

import numpy as np
import math
import random
from train import train, evaluate
import setup_exp
# import neptune.new as neptune
import wandb

import models
import dataset

PROJECT_NAME= 'mutation'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fuzzing Model")
    
    parser.add_argument("--seed", default=1253, type=int,
                        help="Randon seeds.")
    parser.add_argument("--mode", default='single_type', choices=models.MODELS, type=str,
                        help="optimizer")
    parser.add_argument("--dataset", default='single', choices=dataset.DATASET, type=str,
                        help="optimizer")
    parser.add_argument('--debug', default=False, action='store_true', help='debug / not tracking')

    # training configuration
    parser.add_argument("--epochs",  default=200, type=int,
                        help="epochs.")
    parser.add_argument("--batch_size",  default=4096, type=int,
                        help="Batch size.")
    parser.add_argument("--lr",  default=0.001, type=float,
                        help="learning rate")
    parser.add_argument("--optim",  default='Adam', choices=['Adam'], type=str,
                        help="optimizer")
    parser.add_argument("--clip_grad",  default=1., type=float,
                        help="gradient clipping")

    # model configuration
    parser.add_argument("--vocab_size",  default=273, type=int,
                        help="Vocabulary size.")
    parser.add_argument("--embedding_dim",  default=64, type=int,
                        help="Embedding Dimensions.")
    parser.add_argument("--hidden_dim",  default=64, type=int,
                        help="Hidden Dimensions of LSTM.")
    parser.add_argument("--drop_p",  default=0.1, type=float,
                        help="dropout.")
    
    # data configuration:
    parser.add_argument("--src_len",  default=50, type=int,
                        help="length of source sentences(prefix, postfix).")
    parser.add_argument("--trg_len",  default=8, type=int,
                        help="length of target sentence(label-type).")

    parser.add_argument("--data_path",  default='data',  type=str,
                        help="data path")
    parser.add_argument("--output_dir",  default='results',  type=str,
                        help="data path")
                        
    parser.add_argument("--train_interval",  default=50, type=int,
                        help="iteration inerval for train.")
    parser.add_argument("--valid_interval",  default=5, type=int,
                        help="epoch inerval for validation.")
    parser.add_argument("--num_workers",  default=2, type=int,
                        help="number of process.")
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_exp.fix_seeds(args.seed)

    if not args.debug:
        setup_exp.make_workspace(args, 'output_dir', 'mode-lr-batch_size')
        try:
            wandb_logger = wandb.init(name=os.path.basename(args.output_dir),
                                config=args, 
                                project=PROJECT_NAME)
        except ModuleNotFoundError:
            wandb_logger = None
    else:
        setup_exp.make_workspace(args, 'output_dir', '')
        wandb_logger = None
                
    
    model = models.load_model(
        args,
        mode=args.mode,
        vocab_size=args.vocab_size, 
        embedding_dim=args.embedding_dim, 
        hidden_dim=args.hidden_dim,
    )
    models.init_weights(model)

    model = torch.nn.DataParallel(model).to(args.device)
    
    # load_dataset
    trainset, validset, testset = dataset.load_dataset(args, args.dataset)


    # for train
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers, drop_last = True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers, drop_last = True)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    train(args, model, optimizer, trainloader, validloader, scheduler, wandb_logger)

    # for test 
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best.pt')))

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers, drop_last = False)
    
    test_stats = evaluate(args, model, testloader)
    if logger is not None:
                logger.log({
                    **{f"test/{k}":v for k,v in valid_stats.items()}
                })  
