from utils.util_methods import fasta_parser, write_fasta
import pandas as pd
import numpy as np

ids_test, seqs_test = fasta_parser('test/seqs.fasta')
df = pd.read_csv('test/labels.csv')
cls = df.columns
labels_test = np.array(df.values)

# 0 表示正样本
for i, c in enumerate(cls):
    new_ids = []
    for j, seq in enumerate(seqs_test):
        if labels_test[j][i] == 1:
            new_ids.append('>' + str(j) + '|0')
        else:
            new_ids.append('>' + str(j) + '|1')

    write_fasta(f'formated_test/{c}T.txt', new_ids, seqs_test)