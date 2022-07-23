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
    "single_refine": "unique-all"
}
DATASET = list(__dict__.keys())



class SingleRefine(Dataset):
    def __init__(self, data_dir, length=50, split='train', cache=None, train_cases=[0,1,2]):

        self.length = length
        self.split = split
        # data_dir = os.path.join(data_dir, __dict__["single_refine"])
        
        if cache is not None and os.path.exists(cache):
            print(f"load dataset from {cache}...")
        
            with open(cache, 'rb') as f:
                self.data = pickle.load(f)

        else:
            path = os.path.join(data_dir, f'train-{__dict__["single_refine"]}') if split == 'train' \
                            else os.path.join(data_dir, f'test-{__dict__["single_refine"]}')

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

        # print(self.data)
        


# class SingleTokenDataset(Dataset):
#     def __init__(self, path = '', length = 0, interval = 20, split='train', cache=None):
#         self.path = path
#         self.length = length
#         self.data = []
        
#         if cache is not None and os.path.exists(cache):
#             print(f"load dataset from {cache}...")
        
#             with open(cache, 'rb') as f:
#                 self.data = pickle.load(f)
#         else:
#             print("build dataset...")
#             self.build_dataset(split)
#             with open(cache, 'wb') as f:
#                 pickle.dump(self.data, f)

#     def build_dataset(self, split):

#         if split == 'train':
#             # files = glob.glob(os.path.join(path, 'datapoints-50-rev*.gz'))
#             files = glob.glob('datapoints-50-rev-train*.gz')
#         else:
#             files = glob.glob('datapoints-50-rev-eval*.gz')
            
#         # print(files)
#         for file in files:
#             with gzip.open(file) as f:
#                 for i, line in enumerate(f):
#                     if i > 10000: break
#                     line = json.loads(line.decode())
#                     # print(type(line),line.keys())
#                     try:         
#                         prefix = line['prefix']
#                         postfix = line['postfix']
#                         label_type = line['label-type']
#                         label_prefix = line['label-prefix']
#                         label_postfix = line['label-postfix']
                        
#                         postfix.reverse()
#                         label_postfix[0].reverse()
#                         self.data.append([prefix, postfix, label_prefix[0], label_postfix[0]])
#                         # self.data[-1]
#                     except Exception as e:
#                         print('Type Error for data, Please check your data')
#                         print(e, line)
#                         continue
#         # print(self.data)

#     def __len__(self):
#         return len(self.data)
    
    
#     def __getitem__(self,idx):
#         pre, post, label_prefix, label_postfix = self.data[idx]
#         # return torch.LongTensor(pre), torch.LongTensor(post), torch.FloatTensor(label_prefix), torch.FloatTensor(label_postfix)
#         return torch.LongTensor(pre[-10:]), torch.LongTensor(post[-10:]), torch.FloatTensor(label_prefix[-10:]), torch.FloatTensor(label_postfix[-10:])



# class SingleDataset(Dataset):

#     def __init__(self, path = '', length = 10, interval = 20, split='train'):
#         self.path = path
#         self.length = length
#         self.data = []
        
        
#         if split == 'train':
#             files = glob.glob(os.path.join(path, 'datapoints-20-*.gz'))[:3]
#         else:
#             files = [glob.glob(os.path.join(path, 'datapoints-20-*.gz'))[-1]]
#         for file in files:
#             with gzip.open(file) as f:
#                 for i, line in enumerate(f):
#                     # if split == 'train' and i >20000: break                    
#                     # elif split in ['valid', 'test'] and i > 1000: break
#                     line = line.decode()
                    
#                     try:         
#                         temp= list(map(int,re.sub(r'[^0-9,]','',line).split(',')))
#                         prefix, postfix, label_type, label_prefix, label_postfix = temp[:interval], temp[interval:interval+20], temp[interval+20], temp[interval+21:interval+41], temp[interval+41:]
#                         postfix.reverse()
#                         self.data.append([prefix, postfix, [label_type]])

#                     except Exception as e:
#                         print('Type Error for data, Please check your data')
#                         print(e, line)
#                         continue
                
#     def __len__(self):
#         return len(self.data)
    
    
#     def __getitem__(self,idx):
#         pre, post, label = self.data[idx]
#         return torch.LongTensor(pre), torch.LongTensor(post), torch.LongTensor(label)


# class Seq2SeqDataset(Dataset):

#     def __init__(self, args, path = 'data', length = 10, interval = 20, split='train'):
#         self.path = path
#         self.length = length
#         self.data = []
        
#         # d = 'datapoints-seq50'
#         d = 'end' 
#         if os.path.exists(os.path.join(path, f'{d}-{split}.pkl')):
#             with open(os.path.join(path, f'{d}-{split}.pkl'), 'rb') as f:
#                 self.data = pickle.load(f)

#         else:
#             if split == 'train':
#                 file = os.path.join(path, f'{d}-train.gz')
#             elif split == 'valid':
#                 file = os.path.join(path, f'{d}-valid.gz')
#             elif split == 'test':
#                 file = os.path.join(path, f'{d}-test.gz')
#             else:
#                 raise Exception("undefined dataset")
            

#             with gzip.open(file) as f:
#                 for i, line in enumerate(tqdm.tqdm(f)):

#                     # if split == 'train' and i >20000: break                    
#                     # elif split in ['valid', 'test'] and i > 1000: break
                    
#                     line = eval(line.decode())
#                     prefix, postfix, label_type = line['prefix'], line['postfix'], line['label-type']

#                     # post reverse
#                     postfix.reverse()

#                     self.data.append([prefix, postfix, [label_type]])

#             with open(os.path.join(path, f'{d}-{split}.pkl'), 'wb') as f:
#                 pickle.dump(self.data, f)
        

        
                
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self,idx):
#         """
#         :return pre, post: (bsz, len_seq)
#         :return label: (bsz, len_label)
#         """
#         pre, post, label = self.data[idx]
#         return torch.LongTensor(pre), torch.LongTensor(post), torch.LongTensor(label)

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


        
        