from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from utils.metrics import instances_overall_metrics, label_overall_metrics, binary_metrics


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def jaccard(labels, threshold=0.1):
    """
    根据label计算jaccard相似度
    """
    label_values = labels.cpu().numpy()
    batch_size = label_values.shape[0]

    mask = torch.zeros((batch_size, batch_size)).float()
    weights = torch.zeros((batch_size, batch_size)).float()

    jcd = np.zeros((batch_size, batch_size))
    for i in range(batch_size):
        for j in range(batch_size):
            ins = 0
            uni = 0
            for k in range(label_values.shape[1]):
                ins += (label_values[i, k] == 1 and label_values[j, k] == 1)
                uni += (label_values[i, k] == 1 or label_values[j, k] == 1)
            w = ins / uni
            if w >= threshold:
                mask[i, j] = 1
                weights[i, j] = w

    return mask, weights

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def evaluation(opt, y_true, y_pred, tag='val'):
    """
    Evaluate the predictive performance
    """
    # print(f"y shape {y_pred.shape} {y_true.shape}") # y shape (1023, 15) (1023, 15)

    df_b = binary_metrics(y_pred, y_true, opt.funcs, 0.5,
                   f'{opt.result_folder}/{tag}_binary.csv', show=False)
    df_i = instances_overall_metrics(np.array(y_pred), np.array(y_true), 0.5,
                                f'{opt.result_folder}/{tag}_sample.csv', show=False)
    df_l = label_overall_metrics(np.array(y_pred), np.array(y_true), 0.5,
                            f'{opt.result_folder}/{tag}_label.csv', show=False)
    return df_b, df_i, df_l


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
