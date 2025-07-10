from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml

import tensorboard_logger as tb_logger
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

    # model dataset
    parser.add_argument('--pretrained_model', type=str, default='./save/SupCon/datasets/_models/tape_lr_0.01_decay_0_bsz_128_temp_0.07_c_0.1_trial_abde/ckpt_epoch_300.pth')
    parser.add_argument('--model', type=str, default='rtape')
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
    parser.add_argument('--trial', type=str, default='0.07_abde',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    with open("config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    opt.funcs = cfg['pts']
    opt.n_class = len(opt.funcs)

    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

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

    trainset = TapeData("iupac", train_seqs, train_labels)
    valset = TapeData("iupac", val_seqs, val_labels)
    testset = TapeData("iupac", test_seqs, test_labels)

    train_loader = DataLoader(trainset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=True, collate_fn=trainset.collate_fn, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=False, collate_fn=valset.collate_fn, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=False, collate_fn=testset.collate_fn, pin_memory=True)
    return train_loader, val_loader, test_loader


def set_model(opt):
    # model = SupConResNet(name=opt.model)
    # criterion = SupConLoss(temperature=opt.temp)
    # 加载预训练模型
    ckpt = torch.load(opt.pretrained_model)
    pretrained_model_dict = ckpt['model'] 

    base_net = TapeNet.from_pretrained('bert-base')
    base_net.load_state_dict(pretrained_model_dict)

    # 加载下游模型，替换bert
    
    model = TapeNetMLC.from_pretrained('bert-base')
    model_dict = model.state_dict()
    st = {}
    for k, v in base_net.named_parameters():
        if k.startswith('bert') and k in model_dict.keys():
            st[k] = v

    model_dict.update(st)
    model.load_state_dict(model_dict)        

    criterion = torch.nn.BCELoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def set_optimizer(opt, model):

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)
    return optimizer


def freeze_layers_bert(model):
    """
    Freeze the bert 
    """
    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False
        else:
            param.requires_grad = True

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_ids = data['input_ids']
        input_mask = data['input_mask']
        targets = data['targets']   # (B, C)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda(non_blocking=True)
            input_mask = input_mask.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        bsz = targets.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(input_ids, input_mask) # (2B, H)
        loss = criterion(features, targets.float())
        # # update metric
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

def validate(val_loader, model, criterion, epoch, opt, tag='val'):
    """one epoch training"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    preds = []
    trues = []
    with torch.no_grad():
        for idx, data in enumerate(val_loader):

            input_ids = data['input_ids']
            input_mask = data['input_mask']
            targets = data['targets']   # (B, C)

            if torch.cuda.is_available():
                input_ids = input_ids.cuda(non_blocking=True)
                input_mask = input_mask.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            bsz = targets.shape[0]

            # compute loss
            pred = model(input_ids, input_mask) # (2B, H)

            loss = criterion(pred, targets.float())
            # metric
            # # update metric
            losses.update(loss.item(), bsz)
            preds.extend(pred.cpu().detach().numpy())
            trues.extend(targets.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)

        df_b, df_i, df_l = evaluation(opt, np.array(trues), np.array(preds), tag + '_' +str(epoch))
        acc = df_i['Accuracy'].iloc[0]

        # print info
        print('Train: [{0}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch,batch_time=batch_time,
                loss=losses))
        sys.stdout.flush()

    return losses.avg, acc

def main():

    opt = parse_option()

    # set_seed(123)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    best_acc = 0
    best_model = None
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        val_loss, acc = validate(val_loader, model, criterion, epoch, opt, 'val')

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('train loss {:.3f}, val loss {:.3f}'.format(loss, val_loss))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            validate(test_loader, model, criterion, '0', opt, 'test')

    # save the best model
    save_file = os.path.join(
        opt.save_folder, 'best.pth')
    save_model(best_model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
