# @Time : 2024/10/25 14:27
from DatasetLoad import *
import numpy as np
import torch
import random
import shutil
from Models import *
from torch.utils.data import TensorDataset, DataLoader

net = CombinedModel()

dataset = DatasetLoad('femnist',0)

assigned_train_ds=TensorDataset(torch.tensor(dataset.train_data),torch.tensor(dataset.train_label))

train_dl = DataLoader(dataset=assigned_train_ds, batch_size=64, shuffle=True)

for data,label in train_dl:
    pred = net(data)
    print(pred.shape)
    break







