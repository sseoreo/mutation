import pickle
import numpy as np

with open("/home/sseo/mutation/mutation3/data/test-unique-july11.pkl", "rb") as f:
    data = pickle.load(f)


accumulate = None
accumulate01 = np.zeros(2)
for prefix, postfix, label_type, label_prefix, label_postfix, case in data:
    label_prefix = np.array(label_prefix[0])
    sum0 = (label_prefix==0).sum()
    accumulate01[0] += sum0
    accumulate01[1] += len(label_prefix)-sum0

    # print(label_prefix[0])
    if accumulate is None:
        accumulate = label_prefix
    else:
        accumulate += label_prefix


dist_uniform = ((accumulate/accumulate.sum(-1) - np.ones(accumulate.shape[0])/accumulate.sum(-1))**2).mean()
print( dist_uniform)
print(accumulate01/accumulate01.sum(-1))

# print(list(data.keys()))