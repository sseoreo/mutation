import os
import json
from shutil import ExecError
from torch.utils.data import Dataset, DataLoader
# from datasets import DatasetDict, load_dataset, load_from_disk, interleave_datasets

import numpy as np
import torch
import re
import gzip
import glob
import tqdm
import pickle
import json


__dict__ = {
    "single": "datapoints-50-rev",
    "seq2seq": "datapoint-seq50",
    
    "july11": {'train': "11July/train-unique-all",
                'eval': "11July/test-unique-all"},

    "july22": {'train': "22July/train_unique",
                'validation': "22July/test_unique",
                'test': "22July/test_unique"}
}
DATASET = list(__dict__.keys())


def load_dataset(args, dataset):

    if args.dataset == 'july11':
        trainset = SingleRefine( args.data_path,
                            dataset=args.dataset,
                            length=args.src_len, 
                            split='train', 
                            train_cases=[0,1,2] if "type" in args.mode else [2],
                            cache='data/train-unique-july11.pkl' \
                                if not args.debug else 'data/test-unique-july11.pkl')

        evalset = SingleRefine( args.data_path, 
                            dataset=args.dataset,
                            length=args.src_len, 
                            split='eval',
                            train_cases=[0,1,2] if "type" in args.mode else [2],
                            cache='data/test-unique-july11.pkl')

        eval_size = len(evalset)
        validset, testset = torch.utils.data.random_split(evalset, (eval_size//2 , eval_size-eval_size//2))
        print(len(trainset), len(validset), len(testset))
        

    elif args.dataset == 'july22':
        trainset = SingleRefine(args.data_path,
                            dataset=args.dataset,
                            length=args.src_len, 
                            split='train', 
                            train_cases=[0,1,2] if "type" in args.mode else [2],
                            cache='data/train-unique-july22.pkl' \
                                if not args.debug else 'data/dev-unique-july22.pkl')

        validset = SingleRefine(args.data_path, 
                            dataset=args.dataset,
                            length=args.src_len, 
                            split='validation',
                            train_cases=[0,1,2] if "type" in args.mode else [2],
                            cache='data/dev-unique-july22.pkl')

        testset = SingleRefine(args.data_path, 
                            dataset=args.dataset,
                            length=args.src_len, 
                            split='test',
                            train_cases=[0,1,2] if "type" in args.mode else [2],
                            cache='data/test-unique-july22.pkl')
        print(len(trainset), len(validset), len(testset))

    elif args.dataset == 'single':
        trainset = SingleToken(args.data_path,
                            length=args.src_len, 
                            split='train', 
                            cache='data/datapoints-50-rev-train.pkl' \
                                if not args.debug else 'data/datapoints-50-rev-eval.pkl')

        evalset = SingleToken(args.data_path, length=args.src_len, split='eval', cache='data/datapoints-50-rev-eval.pkl')
        eval_size = len(evalset)
        validset, testset = torch.utils.data.random_split(evalset, (eval_size//2 , eval_size-eval_size//2))
        print(len(trainset), len(validset), len(testset))

    elif args.dataset == 'seq2seq':
        trainset = Seq2SeqToken(args.data_path, 
                            length=args.src_len, 
                            split='train', 
                            cache='data/datapoint-seq50-train.pkl'\
                                if not args.debug else 'data/datapoint-seq50-eval.pkl')
        evalset = Seq2SeqToken(args.data_path, length=args.src_len, split='eval', cache='data/datapoint-seq50-eval.pkl')
        eval_size = len(evalset)
        validset, testset = torch.utils.data.random_split(evalset, (eval_size//2 , eval_size-eval_size//2))
        print(len(trainset), len(validset), len(testset))

    else:
        raise Exception("Not defined mode!")

    return trainset, validset, testset 


class SingleRefine(Dataset):
    def __init__(self, data_dir, dataset, length=50, split='train', cache=None, train_cases=[0,1,2]):

        self.length = length
        self.split = split
        # data_dir = os.path.join(data_dir, __dict__["single_refine"])
        
        if cache is not None and os.path.exists(cache):
            print(f"load dataset from {cache}...")
        
            with open(cache, 'rb') as f:
                self.data = pickle.load(f)

        else:
            path = os.path.join(data_dir, __dict__[dataset][split])

            self.data = self.build_dataset(path)
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.data, f)

        self.data = list(filter(lambda x: x[-1] in train_cases, self.data))
        
    def __getitem__(self,idx):
        pre, post, label_type, label_prefix, label_postfix, case = self.data[idx]
        # print(pre, post, label_type, label_prefix, label_postfix, case)
        # return torch.LongTensor(pre), torch.LongTensor(post), torch.FloatTensor(label_prefix), torch.FloatTensor(label_postfix)
        return (
            torch.LongTensor(pre[-1*self.length:]), 
            torch.LongTensor(post[-1*self.length:]), 
            torch.LongTensor(label_type), 
            torch.LongTensor(label_prefix[-1*self.length:]), 
            torch.LongTensor(label_postfix[-1*self.length:]),
                )

    def __len__(self):
        return len(self.data)
    
    
    def build_dataset(self, path):
        data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                
                line = json.loads(line)
                prefix = line['prefix']
                postfix = line['postfix']
                label_type = line['label-type']
                label_prefix = line['label-prefix']
                label_postfix = line['label-postfix']
                case = line['case']
                postfix.reverse()
                label_postfix[0].reverse()
                data.append([prefix, postfix, label_type, label_prefix, label_postfix, case])
        return data

        # print(self.data)

class SingleToken(Dataset):
    def __init__(self, data_dir, dataset='single', length=50, split='train', cache=None):
        self.length = length
        self.split = split
        data_dir = os.path.join(data_dir, __dict__[dataset])
        
        if cache is not None and os.path.exists(cache):
            print(f"load dataset from {cache}...")
        
            with open(cache, 'rb') as f:
                self.data = pickle.load(f)

        else:
            path = f"{data_dir}-train.gz" if split == 'train' else f"{data_dir}-eval.gz"
            self.data = self.build_dataset(path)
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.data, f)
    

    def __getitem__(self,idx):
        pre, post, label_type, label_prefix, label_postfix = self.data[idx]
        # return torch.LongTensor(pre), torch.LongTensor(post), torch.FloatTensor(label_prefix), torch.FloatTensor(label_postfix)
        return (
            torch.LongTensor(pre[-1*self.length:]), 
            torch.LongTensor(post[-1*self.length:]), 
            torch.LongTensor(label_type), 
            torch.LongTensor(label_prefix[-1*self.length:]), 
            torch.LongTensor(label_postfix[-1*self.length:])
                )

    def __len__(self):
        return len(self.data)
    
    
    def build_dataset(self, path):
        data = []
        with gzip.open(path) as f:
            for i, line in enumerate(f):
                
                line = json.loads(line.decode())
                prefix = line['prefix']
                postfix = line['postfix']
                label_type = line['label-type']
                label_prefix = line['label-prefix']
                label_postfix = line['label-postfix']
                postfix.reverse()
                label_postfix[0].reverse()
                data.append([prefix, postfix, label_type, label_prefix[0], label_postfix[0]])
        return data

        # print(self.data)
        
class Seq2SeqToken(Dataset):
    def __init__(self, data_dir, length=50, split='train', cache=None):
        self.length = length
        self.split = split
        data_dir = os.path.join(data_dir, __dict__['seq2seq'])
        
        if cache is not None and os.path.exists(cache):
            print(f"load dataset from {cache}...")
        
            with open(cache, 'rb') as f:
                self.data = pickle.load(f)

        else:
            path = f"{data_dir}-train.gz" if split == 'train' else f"{data_dir}-eval.gz"
            self.data = self.build_dataset(path)
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump(self.data, f)

    

    def __getitem__(self,idx):
        pre, post, label_type, label_prefix, label_postfix = self.data[idx]
        # return torch.LongTensor(pre), torch.LongTensor(post), torch.FloatTensor(label_prefix), torch.FloatTensor(label_postfix)
        # print(torch.LongTensor(label_prefix).shape)
        return (
            torch.LongTensor(pre[-1*self.length:]), 
            torch.LongTensor(post[-1*self.length:]), 

            # torch.LongTensor(post[:self.length]), 
            torch.LongTensor(label_type), 
            torch.LongTensor(label_prefix)[:, -1*self.length:], 
            torch.LongTensor(label_postfix)[:, -1*self.length:]
                )

    def __len__(self):
        return len(self.data)
    
    
    def build_dataset(self, path):
        data = []
        # print(path)
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                # print(line, i)
                line = json.loads(line)

                prefix = line['prefix']
                postfix = line['postfix']
                label_type = line['label-type']
                label_prefix = line['label-prefix']
                label_postfix = line['label-postfix']
                postfix.reverse()
                [l.reverse() for l in label_postfix]
                # print(label_type, postfix, label_postfix)
                data.append([prefix, postfix, label_type, label_prefix, label_postfix])
        return data



if __name__ =='__main__':       
    # dataset = SingleToken('data', length=5, split='eval', cache='data/datapoints-50-rev-eval.pkl')

    dataset = Seq2SeqToken('data', length=5, split='eval', cache='data/datapoint-seq50-eval.pkl')
    # temp = dataset.__getitem__(3)  
    # print(temp)

    loader = DataLoader(dataset, batch_size = 1, shuffle = False)
    for i, (pre, post,label_type, label_pre, label_post) in enumerate(loader):
        if i <2:
            print(pre, post, label_type, label_post)
            # print(pre.shape, post.shape ,label_type.shape, label_pre.shape, label_post.shape)


        
        