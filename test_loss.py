from util import set_optimizer, save_model, jaccard
import numpy as np
import torch
from losses import MlcSupConLoss

# 测试jaccard
vec1 = torch.tensor([[0.5, 0.5, 0.5, 0], [0.1, 0.1, 0.1, 0], [0.7, 0.7, 0.7, 0]], dtype=torch.float)
vec2 = torch.tensor([[0.5, 0.5, 0.5, 1], [0.1, 0.1, 0.1, 1], [0.7, 0.7, 0.7, 1]], dtype=torch.float)
features = torch.cat((vec1.unsqueeze(1), vec2.unsqueeze(1)), dim=1)

labels = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]])
labels = torch.from_numpy(labels)
masks, weights = jaccard(labels, 0.5)

print(masks)
print(weights)

# 测试loss
criterion = MlcSupConLoss()
loss = criterion(features, labels, masks, weights)
print(loss.item())