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
from utils.losses import FocalLoss
import random
import numpy as np
from pssmModel import Data,Combine_Dataset,Combine_Dataset_val
from pssmModel import FModel, FModel2
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss
from sklearn.metrics import hamming_loss

def new_instances_overall_metrics(index, y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    计算样本层面的整体评价指标
    """
    y_pred_cls = np.zeros_like(y_pred, dtype=np.int)
    y_pred_cls[y_pred > threshold] = 1    # 预测类别
    n, m = y_true.shape
    # Hamming Loss
    HLoss = hamming_loss(y_true, y_pred_cls)
    # Accuracy
    ACC = 0
    for i in range(n):
        ACC += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum((y_pred_cls[i] == 1) | (y_true[i] == 1)))
    ACC /= n
    # Precision
    Precision = 0
    for i in range(n):
        if (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) == 0): continue
        Precision += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum(y_pred_cls[i] == 1) )
    Precision /= n
    # Recall
    Recall = 0
    for i in range(n):
        Recall += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum(y_true[i] == 1))
    Recall /= n

    # Absolute ture
    AT = 0
    for i in range(n):
        if(np.all(y_pred_cls[i] == y_true[i])):
            AT += 1
    AT /= n
    df = pd.DataFrame({'HLoss': [HLoss], 'Accuracy': [ACC], 'Precision': [Precision], 'Recall': [Recall], 'Absolute true': [AT]})

    if df_all.loc[0,'Accuracy'] < ACC :
        df_all.loc[0,'Accuracy'] = ACC
    #######################################################
    df_all.loc[index, 'HLoss'] = HLoss
    df_all.loc[index, 'Accuracy'] = ACC
    df_all.loc[index, 'Precision'] = Precision
    df_all.loc[index, 'Recall'] = Recall
    df_all.loc[index, 'Recall'] = AT
    #######################################################
    # if show:
    #     print(df)
    # if save is not None:
    #     df_all.to_csv(save)

    return df

"""
固定TAPE, 重新训练MLP
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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')

    # model dataset
    parser.add_argument('--pretrained_model', type=str, default='./save/SupCon/datasets/_models/tape_lr_0.01_decay_0_bsz_128_temp_0.07_c_0.1_trial_abde/ckpt_epoch_300.pth')
    parser.add_argument('--model', type=str, default='orgtape')
    parser.add_argument('--dataset', type=str, default='./datasets/', help='dataset')
    parser.add_argument('--p', type=float, default=0.1, help='parameter for random mutation')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0.07_scratch',
                        help='id for recording multiple runs')
    
    #################################################
    parser.add_argument('--nce_k', type=int, default=20)   # 16384
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--softmax', action='store_true',default=True, help='using softmax contrastive loss rather than NCE')
    ##################################################

    parser.add_argument("--pssm_weight", type=float, default=0.17, help="The Weight of pssm feature")

    ######

    opt = parser.parse_args()

    with open("config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    opt.funcs = cfg['pts']
    opt.n_class = len(opt.funcs)

    opt.model_path = './save_model/{}_{}_{}_models'.format(opt.dataset, opt.learning_rate, opt.pssm_weight)
    opt.tb_path = './save_model/{}_{}_{}_tensorboard'.format(opt.dataset, opt.learning_rate, opt.pssm_weight)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial)

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

    opt.result_folder = os.path.join(opt.model_path, opt.model_name)
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
    # pssm_feas = new_pssm_encoding("features/new_pssm", data='train')
    # pssm_pad_feas = pad_features(pssm_feas)
    # print(np.array(pssm_pad_feas).shape)
    pssm_feas = phsical_encoding(pssm_seqs, data='train')
    pssm_pad_feas = pad_features(pssm_feas)
    # print(np.array(pssm_pad_feas).shape)
    pssm_dataset = Data(pssm_pad_feas)

    pssm_seqsv, pssm_labelsv = load_seqs_and_labels("./datasets/val", opt.funcs)
    pssm_feasv = new_pssm_encoding("features/new_pssm", data='val')
    # pssm_pad_feasv = pad_features(pssm_feasv)
    pssm_feasv = phsical_encoding(pssm_seqsv, data='val')
    pssm_pad_feasv = pad_features(pssm_feasv)
    pssm_datasetv = Data(pssm_pad_feasv)

    pssm_seqst, pssm_labelst = load_seqs_and_labels("./datasets/test", opt.funcs)
    # pssm_feast = new_pssm_encoding("features/new_pssm", data='test')
    pssm_feast = phsical_encoding(pssm_seqst, data='test')
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
                    shuffle=True,  pin_memory=True)
    features_dft_loader=DataLoader(features_dft, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True,  pin_memory=True)
   
        
    n_data = len(trainset)

    
    val_loader = DataLoader(valset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True, pin_memory=True)
    return  val_loader, test_loader,features_df_loader,features_dfv_loader,features_dft_loader,n_data

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)

def set_model(opt,n_data):

    model = FModel()
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
    # enable synchronized Batch Normalization
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        contrast = contrast.cuda()
        cudnn.benchmark = True

    return model, criterion, contrast,criterion_l,criterion_ab

def set_model2(opt,n_data):

    model = FModel2()
    if torch.cuda.is_available():
        model = model.cuda()

    # criterion = torch.nn.BCELoss()
    criterion = FocalLoss()

    # enable synchronized Batch Normalization
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def set_optimizer(opt, model):

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)
    return optimizer


# def freeze_layers_bert(model):
#     """
#     Freeze the bert 
#     """
#     for name, param in model.named_parameters():
#         if name.startswith('bert'):
#             param.requires_grad = False
#         else:
#             param.requires_grad = True

def train(features_df_loader, model, contrast, criterion_l, criterion_ab, optimizer, epoch, opt):
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
        pssm_features=pssm_features*opt.pssm_weight

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

def train2(features_df_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    # pssm_model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, batch_data in enumerate(features_df_loader):
        pssm_features = batch_data['pssm_fea']
        sematic_features = batch_data['seq']
        index = batch_data['index']
        target = batch_data['target']
        pssm_features=pssm_features*opt.pssm_weight
        # sematic_features = np.reshape(sematic_features, (128, -1))
        # pssm_features = np.reshape(pssm_features, (128, -1))
        # sematic_features = sematic_features[:, :128]
        # pssm_features =  pssm_features[:, :128]
        if torch.cuda.is_available():
            pssm_features = pssm_features.cuda()
            sematic_features = sematic_features.cuda()
            target = target.cuda()

        out = model(pssm_features, sematic_features) # pssm, tape
        loss = criterion(out, target.float())
        
        # # # update metric
        # # losses.update(l_loss.item(), bsz)
        losses.update(loss.item())

        # # # SGD
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


def validate_first(features_dft_loader, model, contrast, criterion_l, criterion_ab, epoch, opt, tag='test'):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    preds = []
    trues = []
    with torch.no_grad():
        for idx, data in enumerate(features_dft_loader):

            # input_ids = data['input_ids']
            # input_mask = data['input_mask']
            pssm_features = data['pssm_fea']
            sematict_features = data['seq']
            index = data['index']
            pssm_features=pssm_features*0.1

            if torch.cuda.is_available():
                pssm_features = pssm_features.cuda()
                sematict_features = sematict_features.cuda()
            
            fea_l, fea_ab = model(pssm_features, sematict_features)

            out_l, out_ab = contrast(fea_ab, fea_l,index)

            # bsz = targets.shape[0]

            # compute loss
            l_loss = criterion_l(out_l)
            ab_loss = criterion_ab(out_ab)

            loss = l_loss + ab_loss

            losses.update(loss.item())
            # metric
            # # update metric
            
            # preds.extend(pred.cpu().detach().numpy())
            # trues.extend(targets.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        
        # df_b, df_i, df_l = evaluation(opt, np.array(trues), np.array(preds), tag + '_' +str(epoch))
        # acc = df_i['Accuracy'].iloc[0]/

        # print info
        print('Validate: [{0}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch,batch_time=batch_time,
                loss=losses))
        sys.stdout.flush()

    return losses.avg

def validate_second(features_dfv_loader, model, criterion, epoch, opt, tag='val'):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    preds = []
    trues = []
    with torch.no_grad():
        for idx, data in enumerate(features_dfv_loader):

            # input_ids = data['input_ids']
            # input_mask = data['input_mask']
            pssm_features = data['pssm_fea']
            sematict_features = data['seq']
            index = data['index']
            target = data['target']
            pssm_features=pssm_features*opt.pssm_weight

            if torch.cuda.is_available():
                pssm_features = pssm_features.cuda()
                sematict_features = sematict_features.cuda()
                target = target.cuda()

            out = model(pssm_features, sematict_features)

            loss = criterion(out, target.float())

            losses.update(loss.item())
            # metric
            # # update metric
            
            preds.extend(out.cpu().detach().numpy())
            trues.extend(target.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        
        df_b, df_i, df_l = evaluation(opt, np.array(trues), np.array(preds), tag + '_' +str(epoch))

        # df_i2 = new_instances_overall_metrics(int(epoch)+1, np.array(preds), np.array(trues), 0.5,
                                    # f'{opt.result_folder}/overall_sample.csv', show=False)

        acc = df_i['Accuracy'].iloc[0]
        f1 = df_l['F1'].iloc[0]

        # print info
        # print('Validate: [{0}]\t'
        #         'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
        #         epoch,batch_time=batch_time,
        #         loss=losses))
        with open('new_save_model/exp5_phisical.txt', "a", encoding='utf-8') as f:
            f.write(f'epoch:{epoch} tag: {tag} acc: {acc} f1: {f1}\n')
        sys.stdout.flush()

    return losses.avg, acc, df_b

def train_first():

    opt = parse_option()

    # set_seed(123)

    # build data loader
    val_loader, test_loader ,features_df_loader,features_dfv_loader,features_dft_loader,n_data= set_loader(opt)

    # build model and criterion
    model, criterion, contrast,criterion_l,criterion_ab= set_model(opt,n_data)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    best_acc = 0
    best_model = None
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(features_df_loader, model, contrast, criterion_l, criterion_ab, optimizer, epoch, opt)
        # val_loss = validate_first(features_dfv_loader, model, contrast, criterion_l, criterion_ab, epoch, opt, 'val')
        test_loss = validate_first(features_dft_loader, model, contrast, criterion_l, criterion_ab, epoch, opt, 'test')

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # print('train loss {:.3f}, val loss {:.3f}'.format(loss, val_loss))

        # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            # save_model(model, optimizer, opt, epoch, save_file)
        
        # if acc > best_acc:
        #     best_acc = acc
        #     best_model = model
        #     validate(test_loader, model, criterion, '0', opt, 'test')

    # save the best model
    save_file = os.path.join(
        opt.save_folder, 'best.pth')
    # save_model(best_model, optimizer, opt, opt.epochs, 'save_model/adasyn.pth')

def train_second():

    opt = parse_option()

    # set_seed(123)

    # build data loader
    val_loader, test_loader ,features_df_loader,features_dfv_loader,features_dft_loader,n_data= set_loader(opt)

    # build model and criterion
    model, criterion = set_model2(opt,n_data)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    best_acc = 0
    best_model = None
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        print(opt.learning_rate, opt.pssm_weight)
        # train for one epoch
        time1 = time.time()
        loss = train2(features_df_loader, model, criterion, optimizer, epoch, opt)
        # val_loss, val_acc = validate_second(features_dfv_loader, model, criterion, epoch, opt, 'val')
        val_loss, acc, df_b = validate_second(features_dfv_loader, model, criterion, epoch, opt, 'val')
    
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            validate_second(features_dft_loader, model, criterion, epoch, opt, 'test')
            df_b.to_csv('save_model/ada9.csv')
            # save_model(best_model, optimizer, opt, opt.epochs, './save_model/ada9_fl_lr_0.0002_pssmweight_0.17.pth')
        elif acc > best_acc:
            best_acc = acc
            best_model = model
            validate_second(features_dft_loader, model, criterion, epoch, opt, 'test')
            # save_model(best_model, optimizer, opt, opt.epochs, './save_model/ada9_fl_lr_0.0002_pssmweight_0.17.pth')

    # save the best model
    # save_file = os.path.join(
    #     opt.save_folder, 'best.pth')

# def aaaa():
#     df_all.loc[1,'HLoss'] = 0.01
#     df_all.loc[2,'HLoss'] = 0.02
#     df_all.to_csv('aaaaa.csv', rese)

if __name__ == '__main__':
    df_all = pd.DataFrame(columns=['HLoss', 'Accuracy', 'Precision', 'Recall', 'Absolute true'])
    df_all.loc[0,'Accuracy'] = 0.0001
    # train_first()
    train_second()
