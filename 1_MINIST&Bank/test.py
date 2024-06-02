import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import torch.optim as optimizer
from tensorboardX import SummaryWriter

path = r'.\\data\\BANK\\select-data.csv'
a = pd.read_csv(path)
c = np.array(a)
print(c.shape)
print(type(c))


label = a["Exited"].values.tolist()
label = np.array(label)
print(len(label))
print(label[:5])

data = a.iloc[:, 1 : -1 : 1]
print(data)