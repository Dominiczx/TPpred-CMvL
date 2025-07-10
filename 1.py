import torch 
import torch.nn as nn 





# class SimpleMLP(nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  hid_dim: int,
#                  out_dim: int,
#                  dropout: float = 0.):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#             )

#     def forward(self, x):
#         return self.main(x)


def main(a):
    pass
    # x = torch.zeros(128,50,20)
    # y = torch.zeros(128,50,768)

    # cnn = nn.Sequential(
    #     nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3, stride=1, padding=1),
    #     nn.ReLU()
    # )
    # x = cnn(x)

    

if __name__ == '__main__':
    epoch = "1"
    main(epoch+1)
    # main()