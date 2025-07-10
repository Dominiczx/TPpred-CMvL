import os
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import re
from tape import ProteinBertModel, TAPETokenizer
import yaml
from sklearn.decomposition import PCA

def write_fasta(fn, ids, seqs):
    with open(fn, 'w') as f:
        for i in range(len(ids)):
            if ids[i].startswith('>'):
                f.write(ids[i] + '\n')
            else:
                f.write('>' + ids[i] + '\n')
            f.write(seqs[i] + '\n')

def fasta_parser(fn: str):
    # 加载序列数据
    ids = []
    seqs = []
    id = 0

    with open(fn, 'r') as f:
        lines = f.readlines()
        seq_tmp = ""

        for i, line in enumerate(lines):
            line = line.strip()
            if line[0] == '>':
                id = line.replace('|','_')
                id = id.split(' ')[0]
            elif i < len(lines) - 1 and lines[i+1][0] != '>':
                seq_tmp += line.strip()
            else:
                seq_tmp += line.strip()
                seqs.append(seq_tmp)
                ids.append(id)
                id = 0
                seq_tmp = ""

    return ids, seqs

# one hot 
def onehot_encoding(seqs):
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    encoding_map = np.eye(len(residues))

    residues_map = {}
    for i, r in enumerate(residues):
        residues_map[r] = encoding_map[i]

    res_seqs = []

    for seq in seqs:
        tmp_seq = [residues_map[r] for r in seq]
        res_seqs.append(np.array(tmp_seq))

    return res_seqs

# pssm
def _pssm_seq2fn_dict(pssm_folder, save = 'data/pssm_seq2fn.pkl'):
    """
    将已经得到PSSM的文件总结为一个seq:filename的字典，方便根据seq查找pssm文件
    :param folder:
    """
    if pssm_folder[-1] != '/' : pssm_folder += '/'
    fs = os.listdir(pssm_folder)
    res = {}
    for fn in fs:
        with open(pssm_folder + fn , 'r') as f:
            lines = f.readlines()
            tmp = []
            for line in lines[3:]:
                line = line.strip()
                lst = line.split(' ')
                while '' in lst:
                    lst.remove('')
                if len(lst) == 0:
                    break
                r = lst[1]
                tmp.append(r)
            seq = ''.join(tmp)
            res[seq] = fn

    if save is not None:
        with open(save, 'wb') as f:
            pkl.dump(res, f)
    return res

def _msa_seq2fn_dict(msa_folder, save = None):
    """
    将已经得到MSA的文件总结为一个seq:filename的字典，方便根据seq查找msa(a3m)文件
    """
    if msa_folder[-1] != '/' : msa_folder += '/'
    fs = os.listdir(msa_folder)
    res = {}
    for fn in fs:
        with open(msa_folder + fn , 'r') as f:
            lines = f.readlines()
            seq = lines[1].strip()
            res[seq] = fn
    if save is not None:
        with open(save, 'wb') as f:
            pkl.dump(res, f)

    return res

def phsical_encoding(pssm_seqs, data='test'):
    encoding_map = {
        'A': [0],
        'F': [0],
        'I': [0],
        'M': [0],
        'L': [0],
        'P': [0],
        'V': [0],
        'G': [1],
        'W': [1],
        'C': [2],
        'N': [3],
        'Q': [3],
        'S': [3],
        'T': [3],
        'Y': [3],
        'D': [4],
        'E': [4],
        'K': [5],
        'H': [5],
        'R': [5],
        'X': [6]  # 对于未知字符，使用[6]
    }

    phisical_info = []
    # 遍历序列中的每个字符并编码
    for seq in pssm_seqs:
        # seq = re.sub('[UZOB]', 'X', seq)
        encoded_sequence = []
        for char in seq:
            if char in encoding_map:
                encoded_sequence.append(encoding_map[char])
            else:
                # 如果字符不在映射中，使用[4, 2]表示未知字符
                encoded_sequence.append(encoding_map['X'])
        # print(torch.tensor(encoded_sequence).shape)
        phisical_info.append(np.array(encoded_sequence))
    # 将编码结果转换为一个tensor并返回
    return phisical_info

def new_pssm_encoding(pssm_dir, data='test'):
    pssms = []
    with open(os.path.join(pssm_dir, data + '_pssm.csv'), "r", encoding='utf-8') as f:
        one_pssm = []
        for index, line in enumerate(f):
            if line[0] == '>':
                if index == 0: continue
                pssms.append(np.array(one_pssm))
                one_pssm = []
                continue
            else:
                tmp = line.strip().split(' ')
                tmp = list(map(int, tmp))
                one_pssm.append(tmp)
        pssms.append(np.array(one_pssm))
    return pssms


def pssm_encoding(seqs, pssm_dir, blosum = True):
    """
    比对已生成的PSSM矩阵，如果比对失败，检查是否使用blosum, 若是，则用blosum替代，否则为空
    res列表中的每个元素都是一个NumPy数组,其形状是 (L, D)
    L 是序列的长度，即PSSM矩阵的行数。
    D 是PSSM矩阵的特征维度，即PSSM矩阵的列数。
    """
    if pssm_dir[-1] != '/': pssm_dir += '/'

    global blosum_dict
    if blosum:
        blosum_dict = _read_blosum('data/blosum62.pkl')

    with open('data/pssm_seq2fn.pkl', 'rb') as f:
        pssm_path_dict = pkl.load(f)

    res = []
    for i , seq in enumerate(seqs):

        if seq in pssm_path_dict.keys():
            # pssm
            pssm_fn = pssm_path_dict[seq]
            tmp = _load_pssm(pssm_fn, pssm_dir)
            res.append(np.array(tmp))
        else:
            if blosum:
                enc = _one_blosum_encoding(seq, blosum_dict)
                res.append(np.array(enc))
            else:
                res.append([])
    return res

def _read_blosum(blosum_dir):
    """Read blosum dict and delete some keys and values."""
    with open(blosum_dir, 'rb') as f:
        blosum_dict = pkl.load(f)

    blosum_dict.pop('*')
    blosum_dict.pop('B')
    blosum_dict.pop('Z')
    blosum_dict.pop('X')
    blosum_dict.pop('alphas')

    for key in blosum_dict:
        for i in range(4):
            blosum_dict[key].pop()
    return blosum_dict

def _load_pssm(query, pssm_path):
    """
    :param query: query id
    :param pssm_path: dir saving pssm files
    :return:
    """
    if pssm_path[-1] != '/': pssm_path += '/'
    with open(pssm_path + query, 'r') as f:
        lines = f.readlines()
        res = []
        for line in lines[3:]:
            line = line.strip()
            lst = line.split(' ')
            while '' in lst:
                lst.remove('')
            if len(lst) == 0:
                break
            r = lst[2:22]
            r = [int(x) for x in r]
            res.append(r)
    return res

def _one_blosum_encoding(seq, blosum_dict):
    """
    :param seq: a single sequence
    :return:
    """
    enc = []
    for aa in seq:
        enc.append(blosum_dict[aa])
    return enc

# sematic feature  列表中的每个元素都是一个形状为 (L, D) 的NumPy数组,其中 L 是序列的长度，D 是模型的隐藏单元数（特征的维度）
def sematic_encoding(seqs, model = "bert-base"):
    # load model
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')  
    res = []
    for seq in seqs:
        token_ids_array = np.array([tokenizer.encode(seq)])
        token_ids = torch.tensor(token_ids_array)
        output = model(token_ids)[0].squeeze(0).detach().cpu().numpy()
        res.append(output)
    return res

def load_sematic(semantic_path="/home/xkr/TPpred-Cons/semantic_feature/semantic.npy"):
    # load model
    res = np.load(semantic_path,allow_pickle=True)
    res = res.tolist()
    return res

def get_sematic(seqs, model = "bert-base"):
    # load model
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')  
    res = []
    for seq in seqs:
        token_ids_array = np.array([tokenizer.encode(seq)])
        token_ids = torch.tensor(token_ids_array)
        output = model(token_ids)[0].squeeze(0).detach().cpu().numpy()
        res.append(output)
    # print(type(res[0]))
    # print(torch.from_numpy(res[0]).size())
    res = np.array(res)
    np.save("/home/xkr/TPpred-Cons/semantic_feature/semantic.npy", res)


def cat(*args):
    res = args[0]
    for matrix in args[1:]:
        for i in range(len(matrix)):
            res[i] = np.hstack((res[i], matrix[i]))
    return res

def load_seqs_and_labels(folder_fasta, names):

    ids, seqs = fasta_parser(os.path.join(folder_fasta, "seqs.fasta"))
    df = pd.read_csv(os.path.join(folder_fasta, "labels.csv"))
    df = df[names]
    labels = np.array(df.values, dtype=np.int32)

    return seqs, labels

def pad_features(feas: list, pad_len = 50):
    res = []
    h = feas[0].shape[1]
    for fea in feas:
        m = fea.shape[0] # len
        if m <= pad_len:
            pad_zeros = np.zeros((pad_len - fea.shape[0], h))
            fea = np.vstack((fea, pad_zeros))
            res.append(fea)
        else:
            fea = fea[:pad_len]
            res.append(fea)
    return np.array(res)

def pad_features_3(feas, pad_len = 768):
    res = []
    h = feas[0].shape[1]
    for fea in feas:

        m = fea.shape[1] # len = 20
        
        if m <= pad_len:
            pad_zeros = np.zeros(( 50 , pad_len - fea.shape[1]))
            
            fea = np.concatenate((fea,pad_zeros),axis=1)
            
            res.append(fea)
        else:
            fea = fea[:pad_len]
            res.append(fea)

    return torch.as_tensor(res)

def pad_sequences(sequences, pad_val = 0, pad_len = 50):
    batch_size = len(sequences)
    array = np.full((batch_size, pad_len), 0, dtype=np.int32)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq
    
    return array


# if __name__ == '__main__':
#     with open("config.yaml", 'r') as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)

#     funcs = cfg['pts']
#     seqs, labels = load_seqs_and_labels("./datasets/train", funcs)
#     seqsv, labelsv = load_seqs_and_labels("./datasets/val", funcs)
#     # # sematic features
#     # print(111)
#     # # feas = sematic_encoding(seqs)
#     # get_sematic(seqs)
#     # # pssm features
#     # print(222)
#     res = pssm_encoding(seqs, "features/pssm")
#     pad_res = pad_features(res)
#     print(pad_res.shape)
#     pca = PCA(n_components= 128)
#     x = pad_res.reshape((8190, 1000))
#     x = pca.fit_transform(x)
#     print(x.shape)
    # print(feas[0].size())
    # print(res[0].size())
    # exit(0)
    
    # one-hot features
    # feas = onehot_encoding(seqs)
    # for fea in feas:
    #     print(fea.shape)
    # tensor = torch.randn(128, 50, 20)
    # res = pad_features_3(tensor,768)
    # print(res.shape)
    # print(f"tensor size {pad_features_3(tensor,768)}")

def get_sematic_val(seqs, model = "bert-base"):
    # load model
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')  
    res = []
    for seq in seqs:
        token_ids_array = np.array([tokenizer.encode(seq)])
        token_ids = torch.tensor(token_ids_array)
        output = model(token_ids)[0].squeeze(0).detach().cpu().numpy()
        res.append(output)
    # print(type(res[0]))
    # print(torch.from_numpy(res[0]).size())
    res = np.array(res)
    np.save("/home/xkr/TPpred-Cons/semantic_feature/semantic_val.npy", res)
