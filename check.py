import os
import gzip
import json


# check
# file = "/home/sseo/mutation/data/datapoints-50-rev-eval.gz"
# with gzip.open(file) as f:
#     for i, line in enumerate(f):
#         if i > 10000: break
#         line = json.loads(line.decode())
#         prefix = line['prefix']
#         label_type = line['label-type']
#         label_prefix = line['label-prefix']
#         for t, l in zip(label_type,label_prefix):
#             eq_t = [[ p, t, l[i] ]  for i, p in enumerate(prefix) if l[i]]
#             print(eq_t)



# def split(data_path, new_path, eval_samples=100000):

#     filename = os.path.basename(data_path).split('.')[-2]
#     train_path = os.path.join(new_path, filename + '-train.gz')

#     data = []
#     with gzip.open(data_path) as f:
        
#         f = list(f)
#         # train
#         # with open()
#         for i, line in enumerate(f[:-1*eval_samples]):
#             line = json.loads(line.decode())
#             d = process(line)
#             print(str(d))
#             # data.append(d)
        
#         # eval
#         for i, line in enumerate(f[eval_samples:]):
#             process(line)
        



# def process(line):
#     label_prefix, label_postfix = make_label(line['label-type'], line['prefix'], line['postfix'])
    
#     d = {
#         "prefix": line['prefix'],
#         "postfix": line['postfix'],
#         "label_type": line['label-type'], # List[int]
#         "label-prefix": label_prefix, 
#         "label-postfix": label_postfix
#     }
#     return d




def make_label(target, prefix, postfix):
    pre, post = [], []
    for i, t in enumerate(target):
        # print(t, prefix[i])
        pre.append([int(p==t) for p in prefix] )
        post.append( [int(p==t) for p in postfix]) 
    
    return pre, post


# txt_write = gzip.open("data/datapoint-seq50-train.gz", "wt")
# data = []
# file = "/home/sseo/mutation/data/datapoint-seq50-train.gz"


txt_write = gzip.open("data/datapoint-seq50-eval.gz", "wt")
data = []
file = "/home/sseo/mutation/data/datapoint-seq50-eval.gz"

with gzip.open(file) as f:
    for i, line in enumerate(f):
        # if i > 10000: break
        line = json.loads(line.decode())
        # prefix = line['prefix']
        # label_type = line['label-type']
        # label_prefix = line['label-prefix']

        label_prefix, label_postfix = make_label(line['label-type'], line['prefix'], line['postfix'])

        d = {
            "prefix": line['prefix'],
            "postfix": line['postfix'],
            "label-type": line['label-type'], # List[int]
            "label-prefix": label_prefix, 
            "label-postfix": label_postfix
        }

        
        txt_write.write(f"{d}\n".replace("'", '"'))
    

txt_write.close()


