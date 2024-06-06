import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optimizer
from torch.utils.data import DataLoader, Dataset


cifar100 = datasets.CIFAR100(root='../data', train=True, download=True)