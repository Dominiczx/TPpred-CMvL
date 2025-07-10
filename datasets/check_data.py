from utils.util_methods import fasta_parser, write_fasta
import pandas as pd
import numpy as np
from collections import Counter

ids_test, seqs_test = fasta_parser('test/seqs.fasta')
ids_val, seqs_val = fasta_parser('val/seqs.fasta')
ids_train, seqs_train = fasta_parser('train/seqs.fasta')

labels_test = np.array(pd.read_csv('test/labels.csv').values)
labels_val = np.array(pd.read_csv('val/labels.csv').values)
labels_train = np.array(pd.read_csv('train/labels.csv').values)

print(labels_test.sum(axis=0))
print(labels_val.sum(axis=0))
print(labels_train.sum(axis=0))

print(labels_test.sum(axis=0) + labels_val.sum(axis=0) + labels_train.sum(axis=0))

all = np.vstack((labels_train, labels_val, labels_test))
cnt = Counter(all.sum(axis=1))
print(cnt)
# print(len(ids_train), len(ids_val), len(ids_test))
# print(len(ids_train) + len(ids_val) + len(ids_test))
