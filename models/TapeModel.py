from tape import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP
import torch
import torch.nn as nn
import torch.nn.functional as F


class TapeNet(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.bert = ProteinBertModel(config)
        self.mlp = SimpleMLP(config.hidden_size, 2048, 128)  # 2048
        self.init_weights()

    def forward(self, input_ids, input_mask=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs

        average = torch.mean(sequence_output, dim=1)

        feat = self.mlp(average)    # (B, H)
        
        feat = F.normalize(feat, dim=1)  # (B, H)

        return feat

class TapeNetMLC(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.bert = ProteinBertModel(config)
        self.mlp = nn.Sequential(
            SimpleMLP(config.hidden_size, 2048, 15), # 2048
            nn.Sigmoid()
        )
        self.init_weights()

    def forward(self, input_ids, input_mask=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs

        average = torch.mean(sequence_output, dim=1)

        feat = self.mlp(average)    # (B, H)
        
        feat = F.normalize(feat, dim=1)  # (B, H)

        return feat

# if __name__ == '__main__':
#     print(feat.size())