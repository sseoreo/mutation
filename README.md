<!-- # train-single-type
```bash
$ scripts/train-seq2seq.sh
$ scripts/train-seq2seq-attn.sh
``` -->

# Prepare dataset
```bash
data/
├── 11July/
│   ├── test-unique-all
│   └── train-unique-all
├── 22July/
│   ├── split.py
│   ├── test_unique_split
│   ├── train_unique
│   ├── train_unique_split
│   └── validation_unique
├── datapoints-50-rev-eval.gz
├── datapoints-50-rev.gz
├── datapoints-50-rev-train.gz
├── datapoint-seq50-eval.gz
├── datapoint-seq50.gz
└── datapoint-seq50-train.gz
```

# hyperparameters


# train-single-point
```bash

# binary cross entropy
$ scripts/train-single-point-bce.sh

# multi-class cross entropy
$ scripts/train-single-point-ce.sh

```

# train-single-point-attn
```bash

# binary cross entropy
$ scripts/train-single-point-attn-bce.sh

# multi-class cross entropy
$ scripts/train-single-point-attn-ce.sh

```
