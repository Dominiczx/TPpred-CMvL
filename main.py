from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

#import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, jaccard
from models.TapeModel import TapeNet
from pssmModel import TapeNetPSSM, TapeNetTape, FModel
from losses import SupConLoss, MlcSupConLoss
from dataset import AugData 
from utils.util_methods import *
import random
import numpy as np
import pandas as pd
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
from torch.utils.data import ConcatDataset

from pssmModel import *
from encoding_methods import _load_pssm
from pssmModel import Data,Combine_Dataset


with open("config.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

funcs = cfg['pts']
n_class = len(funcs)

def set_seed(seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,200',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='tape')
    parser.add_argument('--dataset', type=str, default='./datasets/', help='dataset')
    parser.add_argument('--p', type=float, default=0.1, help='parameter for random mutation')
    parser.add_argument('--c', type=float, default=0.1, help='parameter for jaccard')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,  # 0.07
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='abde',
                        help='id for recording multiple runs')


    #################################################
    parser.add_argument('--nce_k', type=int, default=20)   # 16384
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--softmax', action='store_true',default=True, help='using softmax contrastive loss rather than NCE')
    ##################################################

    opt = parser.parse_args()

    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_lr_{}_decay_{}_bsz_{}_temp_{}_c_{}_trial_{}'.\
        format(opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.c, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    """
    load and argument training set
    """
    # @TODO
    
    # 加载数据集
    train_seqs, train_labels = load_seqs_and_labels("./datasets/train", funcs)
    val_seqs, val_labels = load_seqs_and_labels("./datasets/val", funcs)

    trainset = load_sematic('/home/xkr/TPpred-Cons/semantic_feature/semantic.npy')

    # trainset = sematic_encoding(train_seqs)
    # valset = sematic_encoding(val_seqs)

    # pssm features
    pssm_seqs, pssm_labels = load_seqs_and_labels("./datasets/train", funcs)
    pssm_feas = pssm_encoding(pssm_seqs, "features/pssm")
    pssm_pad_feas = pad_features(pssm_feas)
    pssm_dataset = Data(pssm_pad_feas)

    pssm_feas = pad_features(pssm_feas)
    trainset = pad_features(trainset)

    features_df = Combine_Dataset(pssm_feas,trainset,train_labels)
    # #在这里打包，使用pd.DataFrame
    # features_array = list(data.values())
    # features_d = {'pssm_feature': pssm_feas, 'sematic_feature': trainset}
    # features_df = pd.DataFrame(features_dict)
    

    train_loader = DataLoader(trainset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True,  pin_memory=True)

    # val_loader = DataLoader(valset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
    #                 shuffle=False, pin_memory=True)
    
    features_df_loader=DataLoader(features_df, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True,  pin_memory=True)
    
    n_data = len(trainset)

    
     
    with open("config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)



    # Prepare data
    pssm_train_loader = DataLoader(pssm_dataset, batch_size=128, num_workers=8, 
                shuffle=False,  pin_memory=True)


    return train_loader,n_data,pssm_train_loader,features_df_loader


def set_model(opt,n_data):
    model = FModel()
    # tape_model = TapeNetTape.from_pretrained('bert-base')
    # model = TapeNetTape.from_pretrained('bert-base')
    # 在这里加一个model2 通过pssmModel 得到另一个所谓的model
    # Prepare model
    # Prepare model
    # pssm_model = TapeNetPSSM.from_pretrained('bert-base')
    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     pssm_model.encoder = torch.nn.DataParallel(pssm_model.encoder)
        # tape_model = tape_model.cuda()
        # pssm_model = pssm_model.cuda()
        model = model.cuda()

    criterion = MlcSupConLoss(temperature=opt.temp, contrast_mode='all',
                                base_temperature=opt.temp)


    contrast = NCEAverage(opt.feat_dim, n_data ,opt.nce_k, opt.nce_t, opt.nce_m, opt.softmax)
    criterion_l = NCESoftmaxLoss() 
    criterion_ab = NCESoftmaxLoss() 


    if torch.cuda.is_available():
        criterion = criterion.cuda()
        contrast = contrast.cuda()
        cudnn.benchmark = True

    return model, criterion, contrast,criterion_l,criterion_ab


def train(train_loader,pssm_train_loader,features_df_loader, model, contrast, criterion_l, criterion_ab, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    # pssm_model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, batch_data in enumerate(features_df_loader):
        pssm_features = batch_data['pssm_fea']
        sematic_features = batch_data['seq']

        index = batch_data['index']
        # sematic_features = np.reshape(sematic_features, (128, -1))
        # pssm_features = np.reshape(pssm_features, (128, -1))
        # sematic_features = sematic_features[:, :128]
        # pssm_features =  pssm_features[:, :128]
        if torch.cuda.is_available():
            pssm_features = pssm_features.cuda()
            sematic_features = sematic_features.cuda()

        fea_l, fea_ab = model(pssm_features, sematic_features) # pssm, tape
        # 到这一步都是ok的
    
        # targets = batch_data['targets']
        # index = batch_data['index'] # 如果index也在DataFrame中
        # pssm_features_expanded = pssm_features.repeat(1, 1, 39)

        # # input_ids = torch.cat([torch.tensor(pssm_features), torch.tensor(sematic_features)], dim=0)
        # input_ids = torch.cat([pssm_features_expanded, sematic_features], dim=2)
        # # 假设input_mask和index已经在正确的格式和数据类型
        # input_mask = torch.tensor(input_mask)

        if torch.cuda.is_available():
            pssm_features = pssm_features.cuda(non_blocking=True)
            sematic_features = sematic_features.cuda(non_blocking=True)
            # targets = targets.cuda(non_blocking=True)
        '''
        torch.Size([128, 50, 20]) pssm (128,128)   flatten
        torch.Size([128, 50, 768]) seq
        '''
      
        out_l, out_ab = contrast(fea_ab, fea_l,index)   # (b, h, 1)

        
        # l_prob = out_l[:, 0].mean()
        # ab_prob = out_ab[:, 0].mean()

        

        # mask, weights = jaccard(targets, threshold=opt.c)   

        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)

        loss = l_loss + ab_loss

        # print("loss = ", loss)
        
        # # update metric
        # losses.update(l_loss.item(), bsz)
        losses.update(loss.item())

        # # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(features_df_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
        
    

    return losses.avg

def valiate(val_loader, model, criterion, epoch, opt):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for idx, data in enumerate(val_loader):

            input_ids_i = data['input_ids_i']
            input_ids_j = data['input_ids_i']
            input_mask_i = data['input_mask_i']
            input_mask_j = data['input_mask_j']  
            targets = data['targets']   # (B, C)

            input_ids = torch.cat([input_ids_i, input_ids_j], dim=0)
            input_mask = torch.cat([input_mask_i, input_mask_j], dim=0)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda(non_blocking=True)
                input_mask = input_mask.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            bsz = targets.shape[0]

            # compute loss
            features = model(input_ids, input_mask) # (2B, H)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # (B, 2, D)
            mask, weights = jaccard(targets)
            # print(criterion)
            # print(features, targets)
            loss = criterion(features, targets, mask, weights)
            # # update metric
            losses.update(loss.item(), bsz)

        # # measure elapsed time
        batch_time.update(time.time() - end)
    
        # # print info
        print('Train: [{0}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch,batch_time=batch_time,
                loss=losses))
        sys.stdout.flush()

    return losses.avg

def train1():

    set_seed(123)
    
    opt = parse_option()

    # build data loader
    train_loader , n_data , pssm_train_loader ,features_df_loader= set_loader(opt)


    # build model and criterion
    model, criterion,contrast, criterion_l,criterion_ab= set_model(opt,n_data)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
   # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        
        loss = train(train_loader, pssm_train_loader, features_df_loader,model, contrast, criterion_l, criterion_ab, optimizer, epoch, opt)
        
        # val_loss = valiate(val_loader, model, criterion, epoch, opt)
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # print('train loss {:.3f}, val loss {:.3f}'.format(loss, val_loss))

        # tensorboard logger
    #    logger.log_value('train_loss', loss, step=epoch)
    #    logger.log_value('val_loss', val_loss, step=epoch)
    #    logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], step=epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

def train2():
    # (by lhw)
    set_seed(123)
    
    opt = parse_option()

    # build data loader
    train_loader , n_data , pssm_train_loader ,features_df_loader= set_loader(opt)

    # set_model2， 得到model=FModel2(). criterion=BCELoss, contrast, criterion_l,criterion_ab都不需要了
    # build model and criterion
    model, criterion,contrast, criterion_l,criterion_ab= set_model(opt,n_data)

    # 加载第一次训练的模型.pth, base_model
    # 加载下游模型，替换前面几层层(除了MLP外前面的基层)
    # @TODO
    # model_dict = model.state_dict()  
    # st = {}
    # for k, v in base_model.named_parameters():
    #     if k.startswith('front') and k in model_dict.keys():
    #         st[k] = v

    model_dict.update(st)
    model.load_state_dict(model_dict)   
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
   # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        
        loss = train(train_loader, pssm_train_loader, features_df_loader,model, contrast, criterion_l, criterion_ab, optimizer, epoch, opt)
        
        # 应该有，在这里面得到acc
        # val_loss = valiate(val_loader, model, criterion, epoch, opt)
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # print('train loss {:.3f}, val loss {:.3f}'.format(loss, val_loss))

        # tensorboard logger
    #    logger.log_value('train_loss', loss, step=epoch)
    #    logger.log_value('val_loss', val_loss, step=epoch)
    #    logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], step=epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)



if __name__ == '__main__':
    train1()
    # train2()
    # test
