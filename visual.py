from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import save_model, evaluation
from models.TapeModel import TapeNet, TapeNetMLC
from losses import SupConLoss, MlcSupConLoss
from dataset import TapeData 
from utils.util_methods import *
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pssmModel import FModel, FModel2
from pssmModel import Data,Combine_Dataset,Combine_Dataset_val

"""
特征分布可视化
"""

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def set_seed(seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument("--pssm_weight", type=float, default=0.17, help="The Weight of pssm feature")

    # model dataset
    # parser.add_argument('--pretrained_model', type=str, default='./save/SupCon/datasets/_models/tape_lr_0.01_decay_0_bsz_128_temp_0.1_c_0.1_trial_5-5/ckpt_epoch_300.pth')
    # parser.add_argument('--model', type=str, default='./save/SupCon/datasets_0.07/_models/+tape_lr_0.01_decay_0_bsz_128_temp_0.1_c_0.1/ckpt_epoch_300.pth')
    parser.add_argument('--model', type=str, default='/home/czx/workspace/TPpred-Cons/save_model/best1.pth')

    # "+tape_lr_0.01_decay_0_bsz_128_temp_0.1_c_0.1"
    parser.add_argument('--dataset', type=str, default='./datasets/', help='dataset')
    parser.add_argument('--p', type=float, default=0.1, help='parameter for random mutation')

    parser.add_argument('--trial', type=str, default='0.1_5-5',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    with open("config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    opt.funcs = cfg['pts']
    opt.n_class = len(opt.funcs)

    opt.fig_path = './save/SupCon/{}_result'.format(opt.dataset)

    opt.model_name = opt.model.split('/')[-2] 
                        
    opt.save_folder = os.path.join(opt.fig_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.result_folder = os.path.join(opt.fig_path, opt.model_name)
    if not os.path.isdir(opt.result_folder):
        os.makedirs(opt.result_folder)

    return opt


def set_loader(opt):
    """
    load and argument training set
    """

    # 加载数据集
    train_seqs, train_labels = load_seqs_and_labels("./datasets/train", opt.funcs)
    val_seqs, val_labels = load_seqs_and_labels("./datasets/val", opt.funcs)
    test_seqs, test_labels = load_seqs_and_labels("./datasets/test", opt.funcs)

    trainset = load_sematic('/home/xkr/TPpred-Cons/semantic_feature/semantic.npy')
    valset = load_sematic('/home/xkr/TPpred-Cons/semantic_feature/semantic_val.npy')
    # valset = TapeData("iupac", val_seqs, val_labels)
    testset = sematic_encoding(test_seqs)

    # pssm features
    pssm_seqs, pssm_labels = load_seqs_and_labels("./datasets/train", opt.funcs)
    pssm_feas = pssm_encoding(pssm_seqs, "features/pssm")
    pssm_pad_feas = pad_features(pssm_feas)
    pssm_dataset = Data(pssm_pad_feas)

    pssm_seqsv, pssm_labelsv = load_seqs_and_labels("./datasets/val", opt.funcs)
    pssm_feasv = pssm_encoding(pssm_seqsv, "features/pssm")
    pssm_pad_feasv = pad_features(pssm_feasv)
    pssm_datasetv = Data(pssm_pad_feasv)

    pssm_seqst, pssm_labelst = load_seqs_and_labels("./datasets/test", opt.funcs)
    pssm_feast = pssm_encoding(pssm_seqst, "features/pssm")
    pssm_pad_feast = pad_features(pssm_feast)
    pssm_datasett = Data(pssm_pad_feast)
    
    pssm_feas = pad_features(pssm_feas)
    trainset = pad_features(trainset)
    pssm_feasv = pad_features(pssm_feasv)
    valset = pad_features(valset)
    pssm_feast = pad_features(pssm_feast)
    testset = pad_features(testset)

    features_df = Combine_Dataset(pssm_feas,trainset,train_labels, sampler = True)

    features_dfv = Combine_Dataset(pssm_feasv,valset,val_labels)

    features_dft = Combine_Dataset(pssm_feast,testset,test_labels)
    
    features_df_loader=DataLoader(features_df, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True,  pin_memory=True)
    features_dfv_loader=DataLoader(features_dfv, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=False,  pin_memory=True)
    features_dft_loader=DataLoader(features_dft, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=False,  pin_memory=True)
   
        
    n_data = len(trainset)

    train_loader = DataLoader(trainset, batch_size=opt.batch_size, num_workers=opt.num_workers)
    val_loader = DataLoader(valset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True, pin_memory=True)
    return  features_df_loader, features_dfv_loader, features_dft_loader


def set_model(opt):
    # use state dict to load model
    # ckpt = torch.load(opt.model)
    # model_dict = ckpt['model'] 
    # model = FModel2()
    # model.load_state_dict(model_dict)
    
    # use model to load model
    model = torch.load(opt.model)['model']
    # if opt.model_name[0] == 'r' or opt.model_name[:2] == '+r':
    #     model = TapeNetMLC.from_pretrained('bert-base')
    # else:
    #     model = TapeNet.from_pretrained('bert-base')
    # model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True

    for name, module in model.named_modules():
        if name == 'mlp':
            module.register_forward_hook(hook)

    return model

# hook
features = []
outputs = []

def hook(module, input, output):
    features.append(input[0])
    outputs.append(output[0])
    return None


def visual(opt, test_loader, model):
    # 可视化
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    end = time.time()

    preds = []
    trues = []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):

            pssm_features = data['pssm_fea']
            sematict_features = data['seq']
            index = data['index']
            target = data['target']
            pssm_features=pssm_features*opt.pssm_weight

            if torch.cuda.is_available():
                pssm_features = pssm_features.cuda()
                sematict_features = sematict_features.cuda()
                target = target.cuda()

            bsz = target.shape[0]

            # compute loss
            pred = model(pssm_features, sematict_features)

            # metric
            # # update metric
            preds.extend(pred.cpu().detach().numpy())
            trues.extend(target.cpu().numpy())
        print(np.array(preds).shape)

        # measure elapsed time
        batch_time.update(time.time() - end)

        df_b, df_i, df_l = evaluation(opt, np.array(trues), np.array(preds), "visual")
        acc = df_i['Accuracy'].iloc[0]

        # select all trues
        preds = np.array(preds)
        tures = np.array(trues)
        y_pred_cls = np.zeros_like(preds, dtype=int)
        y_pred_cls[preds > 0.5] = 1    # 预测类别
        for i in range(len(y_pred_cls)):
            if(np.all(y_pred_cls[i] == trues[i])):
                # print(i, trues[i], preds[i][0])
                continue
        
        print(pssm_features.shape, sematict_features.shape)
        feat = torch.cat([pssm_features, sematict_features], dim=2)
        feat = feat.flatten()[:660534].reshape(np.array(preds).shape[0], -1)
        print(feat.shape)
        # feat = torch.tensor(preds)
        # exit(0)
        # feat = torch.cat(features, 0)
        feat = feat.cpu().numpy()
        trues = np.array(trues)
        preds = np.array(preds)

        # 特征降维
        # pca = PCA(n_components=2)
        # feat = pca.fit_transform(feat)
        tsne = TSNE(n_components=2, n_iter=400)
        feat = tsne.fit_transform(feat)
        print(feat.shape)

        y0 = [0]
        y1 = [5]
        y2 = [6]
        y3 = [7]

        t0 = np.zeros(15, dtype=np.int64)
        for i in y0: t0[i] = 1
        select_0 = np.all(trues == t0, axis=1)
        s0 = feat[select_0]

        t1 = np.zeros(15, dtype=np.int64)
        for i in y1: t1[i] = 1
        select_1 = np.all(trues == t1, axis=1)
        s1 = feat[select_1]

        t2 = np.zeros(15, dtype=np.int64)
        for i in y2: t2[i] = 1
        select_2 = np.all(trues == t2, axis=1)
        s2 = feat[select_2]

        t3 = np.zeros(15, dtype=np.int64)
        for i in y3: t3[i] = 1
        select_3 = np.all(trues == t3, axis=1)
        s3 = feat[select_3]

        plt.figure(dpi=300)
        plt.scatter(s2[:,1], s2[:,0], c='#FFE699', label='AMP TXP', alpha=0.8)
        plt.scatter(s3[:,1], s3[:,0], c='#A9D18F', label='AIP', alpha=0.8)
        plt.scatter(s0[:,1], s0[:,0], c='#F92A2A', label='AMP', alpha=0.8)
        plt.scatter(s1[:,1], s1[:,0], c='#ED7D31', label='TXP', alpha=0.8)
        plt.legend()  

        plt.savefig(os.path.join('figure', 'test.png'))
        for i in range(0, 15):
            # 找label 为 i 的sample
            t1 = np.zeros(15, dtype=np.int64)
            t1[i] = 1
            select_i = np.all(trues == t1, axis=1)
            si = feat[select_i]

            for j in range(i + 1, 15):
                # 找label 为 j 的sample
                t2 = np.zeros(15, dtype=np.int64)
                t2[j] = 1
                select_j = np.all(trues == t2, axis=1)
                sj = feat[select_j]

                # 找label为i和j的sample
                t3 = np.zeros(15, dtype=np.int64)
                t3[i] = 1
                t3[j] = 1
                select_ij = np.all(trues == t3, axis=1)
                sij = feat[select_ij]
                ni = si.shape[0] 
                nj = sj.shape[0]
                nij = sij.shape[0]
                if ni > 10 and nj > 10 and nij > 10:
                    plt.figure(dpi=300)
                    plt.scatter(si[:,0], si[:,1], c='r')
                    plt.scatter(sj[:,0], sj[:,1], c='b')
                    plt.scatter(sij[:,0], sij[:,1], c='y')
                    plt.savefig(os.path.join('figure/', 'test{}_{}.png'.format(i, j)))
    return 0, acc 



def main():

    opt = parse_option()

    set_seed(123)

    # # build data loader
    # 可视化训练集/测试集特征分布
    train_loader, val_loader, test_loader = set_loader(opt)

    # # build model 
    model = set_model(opt)  # 加载第一阶段训练的模型
    _, acc = visual(opt, train_loader, model)    # 可视化
    
    print("Acc=", acc)


if __name__ == '__main__':
    main()
