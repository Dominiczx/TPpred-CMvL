import numpy as np
import torch
#from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
import argparse
from models.CMC import CMCModel
#from models.GAT import GATModel
from utils.data_processing_2 import load_data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import time
import datetime
import warnings
import torch
from scipy.sparse import *
warnings.filterwarnings("ignore")

def train(args):
    threshold = args.d

    # loading and spliting data
    if args.pos_v == "" or args.neg_v == "":
        
        fasta_path_positive = args.pos_t
        npz_dir_positive = args.pos_npz
        data_list, labels = load_data(fasta_path_positive, npz_dir_positive, threshold, 1)

        fasta_path_negative = args.neg_t
        npz_dir_negative = args.neg_npz

        neg_data = load_data(fasta_path_negative, npz_dir_negative, threshold, 0)
        data_list.extend(neg_data[0])
        labels = np.concatenate((labels, neg_data[1]), axis=0)

        data_train, data_val, _, _ = train_test_split(data_list, labels, test_size=0.25, shuffle=True, random_state=41)
    else:

        fasta_path_train_positive = args.pos_t
        fasta_path_val_positive = args.pos_v
        npz_dir_positive = args.pos_npz
        data_train, _ = load_data(fasta_path_train_positive, npz_dir_positive, threshold, 1)
        data_val, _ = load_data(fasta_path_val_positive, npz_dir_positive, threshold, 1)

        fasta_path_train_negative = args.neg_t
        fasta_path_val_negative = args.neg_v
        npz_dir_negative = args.neg_npz
        neg_data_train, _ = load_data(fasta_path_train_negative, npz_dir_negative, threshold, 0)
        neg_data_val, _ = load_data(fasta_path_val_negative, npz_dir_negative, threshold, 0)

        data_train.extend(neg_data_train)
        data_val.extend(neg_data_val)

        data_train = shuffle(data_train)  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    node_feature_dim = data_train[0]['x'].shape[1]
    node_number = data_train[0]['adj'].shape[1]
    n_class = 2
    #node_num=data_train[0].x.shape[0]

    # tensorboard, record the change of auc, acc and loss
    writer = SummaryWriter()

    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(data_train, batch_size=args.b)
    val_dataloader = DataLoader(data_val, batch_size=args.b)

    if args.pretrained_model == "":
        #model = GATModel(node_feature_dim, args.hd, n_class, args.drop, args.heads).to(device)
        model = CMCModel(node_feature_dim, node_number, args.b, n_class, args.drop, args.hd, args.k).to(device)
    else:
        model = torch.load(args.pretrained_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


    best_acc = 0
    best_auc = 0
    min_loss = 1000
    save_acc = '/'.join(args.save.split('/')[:-1]) + '/acc_' + args.save.split('/')[-1]
    save_auc = '/'.join(args.save.split('/')[:-1]) + '/auc_' + args.save.split('/')[-1]
    save_loss = '/'.join(args.save.split('/')[:-1]) + '/loss_' + args.save.split('/')[-1]

    for epoch in range(args.e):
        print('Epoch ', epoch)
        model.train()
        arr_loss = []
        for i, data in enumerate(train_dataloader):
            #print(data['x'].shape)
            optimizer.zero_grad()
            data['x'] = torch.where(torch.isnan(data['x']), torch.full_like(data['x'], 0), data['x'])
            data['adj'] = torch.where(torch.isnan(data['adj']), torch.full_like(data['adj'], 0), data['adj'])
            data['y'] = torch.where(torch.isnan(data['y']), torch.full_like(data['y'], 0),data['y'])
            data['x'] = data['x'].to(device)
            data['adj'] = data['adj'].to(device)
            output = model(data['x'], data['adj'])
            out = output[0]
            #print(out.shape) #8*2
            #print(data['y'].shape)
            #out= torch.where(torch.isnan(out), torch.full_like(out, 10), out)
            #data.y= torch.where(torch.isnan(data.y), torch.full_like(data.y, 0), data.y)
            data['y'] = data['y'].to(device, dtype=torch.long)
            loss = criterion(out, data['y'])
            loss.backward()
            optimizer.step()
            arr_loss.append(loss.item())
        
        avgl = np.mean(arr_loss)
        print("Training Average loss :", avgl)

        model.eval()
        with torch.no_grad():
            total_num = 0
            total_correct = 0
            preds = []
            y_true = []
            arr_loss = []
            for data in val_dataloader:
                data['x'] = torch.where(torch.isnan(data['x']), torch.full_like(data['x'], 0), data['x'])
                data['adj'] = torch.where(torch.isnan(data['adj']), torch.full_like(data['adj'], 0), data['adj'])
                data['y'] = torch.where(torch.isnan(data['y']), torch.full_like(data['y'], 0),data['y'])
                data['x'] = data['x'].to(device)
                data['adj'] = data['adj'].to(device)
                output = model(data['x'], data['adj'])
                out = output[0]
                #print(out.shape) #8*2
                #print(data['y'].shape)
                #out= torch.where(torch.isnan(out), torch.full_like(out, 10), out)
                #data.y= torch.where(torch.isnan(data.y), torch.full_like(data.y, 0), data.y)
                data['y'] = data['y'].to(device, dtype=torch.long)
                loss = criterion(out, data['y'])
                arr_loss.append(loss.item())

                pred = out.argmax(dim=1)
                score = F.softmax(out, dim=1)[:, 1]
                correct = (pred == data['y']).sum().float()
                total_correct += correct
                total_num += args.b
                preds.extend(score.cpu().detach().data.numpy())
                y_true.extend(data['y'].cpu().detach().data.numpy())

            acc = (total_correct / total_num).cpu().detach().data.numpy()
            auc = roc_auc_score(y_true, preds)
            val_loss = np.mean(arr_loss)
            print("Validation accuracy: ", acc)
            print("Validation auc:", auc)
            print("Validation loss:", val_loss)

            writer.add_scalar('Loss', avgl, global_step=epoch)
            writer.add_scalar('acc', acc, global_step=epoch)
            writer.add_scalar('auc', auc, global_step=epoch)

            if acc > best_acc:
                best_acc = acc
                torch.save(model, save_acc)

            if auc > best_auc:
                best_auc = auc
                torch.save(model, save_auc)

            if np.mean(val_loss) < min_loss:
                min_loss = val_loss
                torch.save(model, save_loss)

            print('-' * 50)

        scheduler.step()

    print('best acc:', best_acc)
    print('best auc:', best_auc)
    if args.o is not None:
        with open(args.o, 'a') as f:
            localtime = time.asctime(time.localtime(time.time()))
            f.write(str(localtime) + '\n')
            f.write('args: ' + str(args) + '\n')
            f.write('auc result: ' + str(best_auc) + '\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument('-pos_t', type=str, default='data/train_data/positive/XU_pretrain_train_positive.fasta',
                        help='Path of the positive training dataset')
    parser.add_argument('-pos_v', type=str, default='',
                        help='Path of the positive validation dataset')
    parser.add_argument('-pos_npz', type=str, default='data/train_data/positive/npz/',
                        help='Path of the positive npz folder, which saves the predicted structure')

    parser.add_argument('-neg_t', type=str, default='data/train_data/negative/XU_pretrain_train_negative.fasta',
                        help='Path of the negative training dataset')
    parser.add_argument('-neg_v', type=str, default='', 
                        help='Path of the negative validation dataset')
    parser.add_argument('-neg_npz', type=str, default='data/train_data/negative/npz/', 
                        help='Path of the positive npz folder, which saves the predicted structure')

    # 0.001 for pretrain， 0.0001 or train
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate') 
    parser.add_argument('-drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-e', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('-b', type=int, default=2, help='Batch size')
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')
    parser.add_argument('-layer', type=int, default=3, help='CNN layer')
    parser.add_argument('-k', type=int, default=1, help='Number of negative samples')

    parser.add_argument('-pretrained_model', type=str, default="",
                        help='The path of pretraining model, if "", the model will be trained from scratch')
    parser.add_argument('-save', type=str, default='saved_models/samp.model',
                        help='The path saving the trained models')
    parser.add_argument('-heads', type=int, default=8, help='Number of heads')

    parser.add_argument('-d', type=int, default=37, help='Distance threshold to construct a graph, 0-37, 37 means 20A')
    parser.add_argument('-o', type=str, default='log.txt', help='File saving the raw prediction results')
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    train(args)

    end_time = datetime.datetime.now()
    print('End time(min):', (end_time - start_time).seconds / 60)
