from tape import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from utils.util_methods import *
from utils.sampling import ImblancedSampling
import torch
from torch.nn import Linear, ReLU, ModuleList, Sequential, Dropout, Softmax, Tanh
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

class SimpleMLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)
    

class FModel(nn.Module):
    def __init__(self):
        super().__init__()
        # (by lhw) 这些MLP是最后一层,你需要在前面加几层CNN或LSTM或Transformer
        # self.cnn_layer_pssm = nn.Sequential(
        #     nn.Conv1d(in_channels=20, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )

        self.cnn_layer_pssm = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )


        self.cnn_layer_tape = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # self.lstm_layer = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)

        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=2048)
        # self.transformer_encoder_stack = nn.TransformerEncoder(self.transformer_encoder, num_layers=6)
        # self.mlp_pssm = SimpleMLP(20, 2048, 128) 
        # self.mlp_tape = SimpleMLP(768, 2048, 128)  

        self.mlp_pssm = SimpleMLP(20, 2048, 128) 
        self.mlp_tape = SimpleMLP(768, 2048, 128) 

# #         self.init_weights()

# #    def init_weights(self):
# #         """ Initialize and prunes weights if needed. """
# #         # Initialize weights
# #         self.apply(self._init_weights)

# #         # Prune heads if needed
# #         if getattr(self.config, 'pruned_heads', False):
# #             self.prune_heads(self.config.pruned_heads)
    def forward(self, x_pssm, x_tape):
        # (by lhw) 这些MLP是最后一层,你需要在前面加几层CNN或LSTM或Transformer
        # x = x_tape
        # if hasattr(self, 'cnn_layer'):
        #     x = self.cnn_layer(x)
        
        # # 通过LSTM层
        # if hasattr(self, 'lstm_layer'):
        #     x, _ = self.lstm_layer(x)
        #     # 假设LSTM输出的是最后一个时间步的隐藏状态
        #     x = x[:, -1, :]
        
        # # 通过Transformer层
        # if hasattr(self, 'transformer_encoder_stack'):
        #     x = self.transformer_encoder_stack(x)

        # print(f"x_pssm {x_pssm.shape} x_tape {x_tape.shape}")  #x_pssm torch.Size([64, 50, 20]) x_tape torch.Size([64, 50, 768])



        x_pssm = x_pssm.float()
        x_tape = x_tape.float()
        
        # x_pssm = x_pssm.squeeze(2)
        # x_tape = x_tape.unsqueeze(2)

        # print(f"x_pssm{x_pssm.shape}, x_tape {x_tape.shape}")

        # exit(0)

    
        # x_pssm = x_pssm.permute(0,2,1)
        # x_tape = x_tape.permute(0,2,1)

        x_pssm = self.cnn_layer_pssm(x_pssm)
        x_tape = self.cnn_layer_tape(x_tape)


        # print(x_pssm.size(), x_tape.size())  -> torch.Size([128, 50, 20]) torch.Size([128, 50, 768])
        x_pssm = self.mlp_pssm(x_pssm)   
        x_tape= self.mlp_tape(x_tape)    

        x_pssm = torch.mean(x_pssm, dim=1)         
        x_tape = torch.mean(x_tape, dim=1)         


        return x_pssm, x_tape

class FModel2(nn.Module):
    def __init__(self):
        super().__init__()
        # (by lhw) 这些MLP是最后一层,你需要在前面加几层CNN或LSTM或Transformer
        # self.cnn_layer_pssm = nn.Sequential(
        #     nn.Conv1d(in_channels=20, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )

        self.cnn_layer_pssm = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )


        self.cnn_layer_tape = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # self.lstm_layer = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)

        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=2048)
        # self.transformer_encoder_stack = nn.TransformerEncoder(self.transformer_encoder, num_layers=6)
        # self.mlp_pssm = SimpleMLP(20, 2048, 128) 
        # self.mlp_tape = SimpleMLP(768, 2048, 128)  

        self.mlp_pssm = SimpleMLP(20, 2048, 15) 
        self.mlp_phisical = SimpleMLP(1, 2048, 15)
        self.mlp_tape = SimpleMLP(768, 2048, 15) 

# #         self.init_weights()

# #    def init_weights(self):
# #         """ Initialize and prunes weights if needed. """
# #         # Initialize weights
# #         self.apply(self._init_weights)

# #         # Prune heads if needed
# #         if getattr(self.config, 'pruned_heads', False):
# #             self.prune_heads(self.config.pruned_heads)
    def forward(self, x_pssm, x_tape):
        # (by lhw) 这些MLP是最后一层,你需要在前面加几层CNN或LSTM或Transformer
        # x = x_tape
        # if hasattr(self, 'cnn_layer'):
        #     x = self.cnn_layer(x)
        
        # # 通过LSTM层
        # if hasattr(self, 'lstm_layer'):
        #     x, _ = self.lstm_layer(x)
        #     # 假设LSTM输出的是最后一个时间步的隐藏状态
        #     x = x[:, -1, :]
        
        # # 通过Transformer层
        # if hasattr(self, 'transformer_encoder_stack'):
        #     x = self.transformer_encoder_stack(x)

        # print(f"x_pssm {x_pssm.shape} x_tape {x_tape.shape}")  #x_pssm torch.Size([64, 50, 20]) x_tape torch.Size([64, 50, 768])



        x_pssm = x_pssm.float()
        x_tape = x_tape.float()
        
        # x_pssm = x_pssm.squeeze(2)
        # x_tape = x_tape.unsqueeze(2)

        # print(f"x_pssm{x_pssm.shape}, x_tape {x_tape.shape}")

        # exit(0)

    
        # x_pssm = x_pssm.permute(0,2,1)
        # x_tape = x_tape.permute(0,2,1)

        x_pssm = self.cnn_layer_pssm(x_pssm)
        x_tape = self.cnn_layer_tape(x_tape)


        # print(x_pssm.size(), x_tape.size())  -> torch.Size([128, 50, 20]) torch.Size([128, 50, 768])
        # x_pssm = self.mlp_pssm(x_pssm)
        x_pssm = self.mlp_phisical(x_pssm)
        x_tape= self.mlp_tape(x_tape)    

        x_pssm = torch.mean(x_pssm, dim=1)         
        x_tape = torch.mean(x_tape, dim=1)         
        
        out = (x_pssm + x_tape) / 2
        out = F.sigmoid(out)
        return out

class TapeNetPSSM(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        
        # self.bert = ProteinBertModel(config)
        self.mlp = SimpleMLP(20, 2048, 128)  # Assuming num_pssm_features for PSSM 但我不知道pssm的维度是多少
        self.init_weights()

    def forward(self, input_ids): #pssm fea 可以通过函数_load_pssm得到，但我不知道pssm的路径
        # input_ids: [B, L, H], [64, 50, 20]
        # return: [B, D]
        # 中间过程
        # W1 : [H, E]
        # w2 : [E, D]
        # [B, L, D]
        # mean [B, D]
        # outputs = self.bert(input_ids, input_mask=input_mask)

        # sequence_output, pooled_output = outputs

        # average = torch.mean(sequence_output, dim=1)

        # Concatenate PSSM features with BERT output
        # combined_features = torch.cat((average, pssm_features), dim=1)

        x = self.mlp(input_ids)    # [B, L, H] -> [B, L, D]
        x = torch.mean(x, dim=1)         
        # feat = F.normalize(feat, dim=1)  # (B, H)

        return x
    
class TapeNetTape(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        
        # self.bert = ProteinBertModel(config)
        self.mlp = SimpleMLP(20, 2048, 128)  # Assuming num_pssm_features for PSSM 但我不知道pssm的维度是多少
        self.init_weights()

    def forward(self, input_ids): 
        x = self.mlp(input_ids)    # [B, L, H] -> [B, L, D]
        x = torch.mean(x, dim=1)         
        # feat = F.normalize(feat, dim=1)  # (B, H)
        return x


class Data(Dataset):
    def __init__(self, X):
        super(Data, self).__init__()

        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float)
    

class Combine_Dataset(Dataset): 
    def __init__(self, pssm_fea, seq, label, sampler=False):
        super(Combine_Dataset, self).__init__()
        self.sampler = sampler
        
        if not self.sampler: 
            self.pssm_fea = pssm_fea
            self.seq = seq
            self.label=label
        else:
            self.seq = []
            self.pssm_fea = []
            self.label = []
            index_list = np.array([[i] for i in range(len(seq))], dtype='float64')
            label_per_class = label.T
            # ros = RandomOverSampler(random_state=123)
            # ros = BorderlineSMOTE(kind='borderline-1', n_jobs=-1, random_state=123)
            # ros = SMOTE(random_state=123)
            ros = ADASYN(random_state=123, n_neighbors=5)
            # ros = ImblancedSampling(label, 1/2)
            for l in label_per_class:
                # print(l.shape, seq.shape)
                # exit(0)
                train_inputs_sampled, train_labels = ros.fit_resample(index_list, l)
                self.seq.extend([seq[int(il[0])] for il in train_inputs_sampled])
                self.pssm_fea.extend([pssm_fea[int(il[0])] for il in train_inputs_sampled])
                self.label.extend([label[int(il[0])] for il in train_inputs_sampled])
            

    def __len__(self):
        return len(self.pssm_fea)

    def __getitem__(self, index):
        seq = self.seq[index]
        pssm_fea = self.pssm_fea[index]
        label=self.label[index]
        
        res = {
            'pssm_fea' : pssm_fea,
            'seq': seq,
            'index':index,
            'target': label
        }
        return res
    
class Combine_Dataset_val(Dataset): 
    def __init__(self, pssm_fea, seq ,label ):
        super(Combine_Dataset_val, self).__init__()
        self.pssm_fea = pssm_fea
        self.seq = seq
        self.label=label

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        seq = self.seq[index]
        pssm_fea = self.pssm_fea[index]
        label=self.label[index]

        res = {
            'pssm_fea' : pssm_fea,
            'seq': seq,
            'index':index,
            'target': label
        }
        return res

if __name__ == '__main__':
    import yaml
    from utils.util_methods import *
    from torch.utils.data import DataLoader

    with open("config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    funcs = cfg['pts']
    seqs, labels = load_seqs_and_labels("./datasets/train", funcs)

    # pssm features
    pssm_feas = pssm_encoding(seqs, "features/pssm")
    pad_feas = pad_features(pssm_feas)
    pssm_dataset = Data(pad_feas)
    # Prepare data
    pssm_train_loader = DataLoader(pssm_dataset, batch_size=128, num_workers=8, 
                shuffle=True,  pin_memory=True)


    pssm_sample = pssm_dataset[0]
    # 打印样本的形状
    print(pssm_sample.shape)  #形状是[50,20]，其中50是我自己定义的，通过pad_features这个函数里面的pad_len定义的。


    train_seqs, train_labels = load_seqs_and_labels("./datasets/train", funcs)
    trainset = sematic_encoding(train_seqs)
    train_loader = DataLoader(trainset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                    shuffle=False, collate_fn=trainset.collate_fn, pin_memory=True)
    for batch in train_loader:
        print(batch.shape)
    # train_pad_feas = pad_features(trainset)
    # train_dataset= Data(train_pad_feas)
    # # 获取trainset的第一个样本
    # train_sample = train_dataset[0]
    # # 打印样本的形状
    # print(train_sample.shape)

    # # Prepare model
    # model = TapeNetPSSM.from_pretrained('bert-base')
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model.encoder = torch.nn.DataParallel(model.encoder)
    #     model = model.cuda()
    #     # criterion = criterion.cuda()
    #     cudnn.benchmark = True

    # for idx, data in enumerate(pssm_train_loader):
    #     if torch.cuda.is_available():
    #         data = data.cuda(non_blocking=True)

    #     feat = model(data)
    #     print(feat.size())


    # 第一步，先把两个向量维度代表什么，接着是，每一个sample对应的两个特征都提取出来，f——1[i]代表的是第i个sample对应的特征，
    # 都对好之后，我再把f——1和f——2打包起来，先打包再经过dataloader，（数据字典  可能需要重写dataloaer
    # 第三步我按照每个dataloader过CMC，




