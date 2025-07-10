from torch.utils.data import Dataset
from torch import tensor
import torch
from tape.tokenizers import TAPETokenizer
import numpy as np
from tape.datasets import pad_sequences as tape_pad
from utils.data_augument import *
import random

class Data(Dataset):
    def __init__(self, X, labels, masks, device):
        super(Data, self).__init__()

        self.X = X
        self.y = labels
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return tensor(self.X[index], dtype=torch.int, device=self.device), \
                tensor(self.y[index], dtype=torch.int, device=self.device), \
                tensor(self.masks[index], dtype=torch.int, device=self.device)

class Data2(Dataset):
    def __init__(self, X, Pzd, labels, masks, device):
        super(Data2, self).__init__()

        self.X = X
        self.pzd = Pzd
        self.y = labels
        self.masks = masks
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return tensor(self.X[index], dtype=torch.int, device=self.device), \
                tensor(self.pzd[index], dtype=torch.float, device=self.device), \
                tensor(self.y[index], dtype=torch.int, device=self.device), \
                tensor(self.masks[index], dtype=torch.int, device=self.device)

class TapeData(Dataset):
    def __init__(self, tokenizer, sequences, labels, max_len = 50):
        super(TapeData, self).__init__()

        self.sequences = [x[: max_len] for x in sequences]
        self.labels = labels
        self.tokenizer = TAPETokenizer(vocab=tokenizer)
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        
        # tokenizer
        seq = self.sequences[index]
        token_ids = self.tokenizer.encode(seq)
        input_mask = np.ones_like(token_ids)
        labels = self.labels[index]

        # padding to max_len
        
        ret = {'input_ids': token_ids,
               'input_mask': input_mask,
               'targets': labels}
        return ret

    def collate_fn(self, batch) :

        elem = batch[0]
        batch = {key: [d[key] for d in batch] for key in elem}
        input_ids = torch.from_numpy(tape_pad(batch['input_ids'], 0))
        input_mask = torch.from_numpy(tape_pad(batch['input_mask'], 0))
        targets = torch.from_numpy(np.array(batch['targets']))
        ret = {'input_ids': input_ids,
               'input_mask': input_mask,
               'targets': targets}
        return ret
    

class AugData(Dataset):
    def __init__(self, tokenizer, sequences, labels, max_len = 50, p = 0.1):
        super(AugData, self).__init__()

        self.sequences = [x[: max_len] for x in sequences]
        self.labels = labels
        self.tokenizer = TAPETokenizer(vocab=tokenizer)
        self.max_len = max_len
        self.argfuncs = [replacement_dict, replacement_alanine,
                        global_random_shuffling, local_random_shuffling,
                        sequence_revsersion, sequence_subsampling]
        self.p = p

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        
        seq = self.sequences[index]
        # randomly select an augument method
        method1 = random.randint(0, 3)
        if method1 in [0, 1]:
            seq1 = self.argfuncs[method1](seq, self.p)
        else:
            seq1 = self.argfuncs[method1 + 1](seq)  # 2->3, 3->4

        method2 = random.randint(0, 3)
        if method2 in [0, 1]:
            seq2 = self.argfuncs[method2](seq, self.p)
        else:
            seq2 = self.argfuncs[method2 + 1](seq)
        # seq1 = seq
        # seq2 = seq
        # tokenizer
        token_ids_i = self.tokenizer.encode(seq1)
        token_ids_j = self.tokenizer.encode(seq2)
        
        input_mask_i = np.ones_like(token_ids_i)
        input_mask_j = np.ones_like(token_ids_j)

        labels = self.labels[index]

        # padding to max_len
        
        ret = {'input_ids_i': token_ids_i,  # list[array]
               'input_ids_j': token_ids_j,
               'input_mask_i': input_mask_i,
               'input_mask_j': input_mask_j,
               'index': index,
               'targets': labels}
        # print(ret.keys())
        # exit(0)
        return ret

    def collate_fn(self, batch):

        elem = batch[0]
        # print(elem)
        batch = {key: [d[key] for d in batch] for key in elem}
        # print(batch)
        # exit(0)

        # pad 
        n_batch = len(batch['input_ids_i'])

        input_ids = batch['input_ids_i']
        input_ids.extend(batch['input_ids_j'])
        input_ids_pad = tape_pad(input_ids, 0)

        input_ids_i = torch.from_numpy(input_ids_pad[: n_batch, :])
        input_ids_j = torch.from_numpy(input_ids_pad[n_batch :, :])

        input_mask = batch['input_mask_i']
        input_mask.extend(batch['input_mask_j'])
        input_mask_pad = tape_pad(input_mask, 0)

        input_mask_i = torch.from_numpy(input_mask_pad[: n_batch, :])
        input_mask_j = torch.from_numpy(input_mask_pad[n_batch :, :])

        targets = torch.from_numpy(np.array(batch['targets']))
        
        index = batch['index']

        ret = {'input_ids_i': input_ids_i,
               'input_ids_j': input_ids_j,
               'input_mask_i': input_mask_i,
               'input_mask_j': input_mask_j,
               'targets': targets,
               'index': index}
        return ret