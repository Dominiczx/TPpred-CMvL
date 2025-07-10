from utils.load_data import load_seqs_and_labels
import os
import yaml
from dataset import TapeData
from torch.utils.data import DataLoader
from models.TapeModel import TapeSimple, TapeLabelAttn
import torch
import numpy as np
from utils.metrics import *
import warnings
import random
warnings.filterwarnings("ignore")
import pickle as pkl

def evaluation(y_true, y_pred, tag='val'):
    """
    Evaluate the predictive performance
    """
    # binary_metrics(y_pred, y_true, funcs, 0.5,
    #                 f'{result_folder}/{task_tag}_{tag}_binary.csv', show=False)
    instances_overall_metrics(np.array(y_pred), np.array(y_true), 0.5,
                                f'{result_folder}/{task_tag}_{tag}_sample.csv', show=False)
    # label_overall_metrics(np.array(y_pred), np.array(y_true), 0.5,
    #                         f'{result_folder}/{task_tag}_{tag}_label.csv', show=False)

def mask_by_len(seqs):
    # 根据长度分组
    # < 20, 20 ~ 29, 30 ~ 39, > 40
    group_masks = np.zeros((4, len(seqs)))

    for i, s in enumerate(seqs):
        if(len(s) < 20):
            group_masks[0, i] = 1
        elif len(s) < 30:
            group_masks[1, i] = 1
        elif len(s) < 40:
            group_masks[2, i] = 1
        else:
            group_masks[3, i] = 1            
    
    return group_masks

def set_seed(self, seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


with open("config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
funcs = cfg['pts']
n_class = len(funcs)
result_folder = "results"
task_tag = "TapeLabelAttn"
batch_size = 64

set_seed(46)

# 加载数据集
train_seqs, train_labels = load_seqs_and_labels("./datasets/out90v4/train", funcs)
val_seqs, val_labels = load_seqs_and_labels("./datasets/out90v4/val", funcs)
test_seqs, test_labels = load_seqs_and_labels("./datasets/out90v4/test", funcs)


# PSSM和非PSSM分组
with open("train.pkl", "rb") as f:
    train_tags = pkl.load(f)
with open("val.pkl", "rb") as f:
    val_tags = pkl.load(f)    
with open("test.pkl", "rb") as f:
    test_tags = pkl.load(f)

# train_seqs_f = []
# val_seqs_f = []
# test_seqs_f = []

# print(len(train_tags), len(val_tags), len(test_tags))

# for i, tag in enumerate(train_tags):
#     if tag == 0:
#         train_seqs_f.append(train_seqs[i])
# for i, tag in enumerate(val_tags):
#     if tag == 0:
#         val_seqs_f.append(val_seqs[i])
# for i, tag in enumerate(test_tags):
#     if tag == 0:
#         test_seqs_f.append(test_seqs[i])

# train_seqs = train_seqs_f
# val_seqs = val_seqs_f
# test_seqs = test_seqs_f

val_len_masks = mask_by_len(val_seqs)
test_len_masks = mask_by_len(test_seqs)

train_labels = train_labels
val_labels = val_labels
tes_labels = test_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = TapeData("iupac", train_seqs, train_labels)
valset = TapeData("iupac", val_seqs, val_labels)
testset = TapeData("iupac", test_seqs, test_labels)

train_dataloader = DataLoader(trainset, batch_size=batch_size, num_workers=16, shuffle=True, 
        collate_fn=trainset.collate_fn)
val_dataloader = DataLoader(valset, batch_size=batch_size, num_workers=16, shuffle=False, 
        collate_fn=valset.collate_fn)
test_dataloader = DataLoader(testset, batch_size=batch_size, num_workers=16, shuffle=False, 
        collate_fn=testset.collate_fn)

model = TapeLabelAttn.from_pretrained('bert-base').to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), 3e-6) # 3e-6 for simple

for i, epoch in enumerate(range(100)):
    model.train()
    train_losses = []
    for _, data in enumerate(train_dataloader):
        optimizer.zero_grad()

        input_ids = data['input_ids'].to(device)
        masks = data['input_mask'].to(device)
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        out = model(input_ids, masks)

        loss = criterion(out, targets.float())
        # loss = TapeSimple.criterion(out, targets.float())
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        for _, data in enumerate(val_dataloader):

            input_ids = data['input_ids'].to(device)
            masks = data['input_mask'].to(device)
            targets = data['targets'].to(device)

            out = model(input_ids, masks)
            loss = criterion(out, targets.float())
            # loss = TapeSimple.criterion(out, targets.float())

            val_losses.append(loss.item())
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(out.cpu().detach().numpy())

    print(f"Epoch {i}, train loss:{np.mean(train_losses)}, val loss: {np.mean(val_losses)}" )
    evaluation(np.array(y_true), np.array(y_pred), "val_g")
    evaluation(np.array(y_true)[val_tags == 1], np.array(y_pred)[val_tags == 1], "val_p")
    evaluation(np.array(y_true)[val_tags == 0], np.array(y_pred)[val_tags == 0], "val_b")
    for i in range(4):
        evaluation(np.array(y_true)[val_len_masks[i] == 1], np.array(y_pred)[val_len_masks[i] == 1], f"val_len_{i}") 

    # test
    model.eval()
    test_losses = []
    y_pred = []
    y_true = []
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):

            input_ids = data['input_ids'].to(device)
            masks = data['input_mask'].to(device)
            targets = data['targets'].to(device)

            out = model(input_ids, masks)
            loss = criterion(out, targets.float())
            # loss = TapeSimple.criterion(out, targets.float())

            test_losses.append(loss.item())
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(out.cpu().detach().numpy())
    evaluation(np.array(y_true), np.array(y_pred), "test_g")
    evaluation(np.array(y_true)[test_tags == 1], np.array(y_pred)[test_tags == 1], "test_p")
    evaluation(np.array(y_true)[test_tags == 0], np.array(y_pred)[test_tags == 0], "test_b")
    for i in range(4):
        evaluation(np.array(y_true)[test_len_masks[i] == 1], np.array(y_pred)[test_len_masks[i] == 1], f"test_len_{i}") 

