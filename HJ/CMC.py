import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
#import TRANS_GRAPH
import numpy as np
import torch
import random
import math
from sklearn.preprocessing import MinMaxScaler
"""
Appling pyG lib
"""


class CMCModel(nn.Module):
    def __init__(self, node_feature_dim, node_number, batch_size, output_dim, drop, hidden_dim, k):
        super(CMCModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.batch_size = batch_size
        self.k = k
        #self.bata = nn.Parameter( torch.randint(1, (1,1)).float())
        self.node_number = node_number
        self.lin2 = Linear(node_number, node_feature_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,    # 输入图片的高度
                out_channels=hidden_dim,  # 输出图片的高度
                kernel_size=5,    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=2,        # 给图外边补上0
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,    # 同上
                out_channels=hidden_dim,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,    # 同上
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,    # 输入图片的高度
                out_channels=hidden_dim,  # 输出图片的高度
                kernel_size=5,    # 5x5的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=2,        # 给图外边补上0
            ),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,    # 同上
                out_channels=hidden_dim,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,    # 同上
                out_channels=1,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU()
        )
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2)
        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(int(node_number*(node_number+node_feature_dim)/4), 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(512, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(self.hidden_dim, self.output_dim))
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, adj):
        #print(x.shape)
        x = torch.unsqueeze(x, dim=1)
        x=self.conv1(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x=self.conv2(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x=self.conv3(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        #print(x.shape)
        
        adj = adj.float()
        adj = torch.unsqueeze(adj, dim=1)
        adj=self.conv4(adj)
        adj = F.dropout(adj, p=self.drop, training=self.training)
        adj=self.conv5(adj)
        adj = F.dropout(adj, p=self.drop, training=self.training)
        adj=self.conv6(adj)
        adj = F.dropout(adj, p=self.drop, training=self.training)

        x = torch.squeeze(x, dim=1)
        adj = torch.squeeze(adj, dim=1)

        b=torch.cat((x, adj), 2)
        b = F.relu(b)
        b = self.MaxPool2d(b)
        b = b.view(x.size(0), -1)
        b = self.Classes(b)
        b = torch.unsqueeze(b, dim=2)
        #print(b.shape)
        
        #loss
        loss=0
        loss_1=[]
        loss_2=[]
        for i in range(x.shape[0]):
            z=[]
            A=x[i]
            B=adj[i]
            similarity_matrix = F.cosine_similarity(A.unsqueeze(1), B.unsqueeze(0), dim=2)
            x_adj_i=similarity_matrix.mean().item()
            x_adj_i= math.exp(x_adj_i)
            z.append(x_adj_i)
            neg_list=[random.randint(0,min(x.shape[0]-1,self.batch_size-1))for i in range(self.k)]
            for j in range(len(neg_list)):
                t=neg_list[j]
                B = adj[t]
                similarity_matrix = F.cosine_similarity(A.unsqueeze(1), B.unsqueeze(0), dim=2)
                z_1=similarity_matrix.mean().item()
                z.append(z_1)
            '''
            xmin = min(z)
            xmax = max(z)
            for i_1, m in enumerate(z):
                z[i_1] = (m - xmin) / (xmax - xmin)
            '''
            h_1=math.log(z[0]/sum(z))
            loss_1.append(h_1)
            z=[]
            B=x[i]
            A=adj[i]
            similarity_matrix = F.cosine_similarity(A.unsqueeze(1), B.unsqueeze(0), dim=2)
            x_adj_i=similarity_matrix.mean().item()
            x_adj_i= math.exp(x_adj_i)
            z.append(x_adj_i)
            neg_list=[random.randint(0,min(x.shape[0]-1,self.batch_size-1))for i in range(self.k)]
            for j in range(len(neg_list)):
                t=neg_list[j]
                B = x[t]
                similarity_matrix = F.cosine_similarity(A.unsqueeze(1), B.unsqueeze(0), dim=2)
                z_1=similarity_matrix.mean().item()
                z.append(z_1)
            '''
            xmin = min(z)
            xmax = max(z)
            for i, m in enumerate(z):
                z[i] = (m - xmin) / (xmax - xmin)
            '''
            h_2=math.log(z[0]/sum(z))
            loss_2.append(h_2)
        #print(loss_2)
        loss=-1*(np.mean(loss_1)+np.mean(loss_2))
        #print(loss)
        return b, loss
