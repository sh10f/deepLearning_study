import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import torch.optim as optimizer
from tensorboardX import SummaryWriter


# forwar() <=> 神经网络模块
class Net(nn.Module):
    """An Linear Model inherits from nn.Module

    隐藏层相当于3层： Flatten + 全连接 + 全连接
    全连接层的激活函数为 Relu
    """

    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x



def train():
    """a fixed training flow

    1.训练过程
        1.1 在 optimizer.zero_grad() 梯度初始化的情况下。
        1.2 每个epoch里，按照 batch_size 遍历 dataloader 计算损失值。
        1.3 通过 minibatch 的样本所计算的损失值，对模型 权重 进行梯度下降修改。

        1.4 每个 epoch 结束后，计算 总体的损失

    2. 验证过程
        2.1 需在评估模式下进行，并设置无梯度：
            model.eval()
            with torch.no_grad():
                xxx

        2.2 每个 epoch 结束后，都可以验证
        2.3 验证就是将 val_set 的样本代入 model() 进行 前向传播。将得到的 result 与 label 进行对比

    """
    for epoch in range(num_epochs):
        total_loss = 0.0
        for image, label in minist_train:
            # RuntimeError: Expected floating point type for target with class probabilities, got Long
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = loss_F(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.mean()
        print(f"Train Epoch {epoch}: loss: {total_loss / len(minist_train)}")
        writer.add_scalar('train_loss', total_loss / len(minist_train), epoch)

        # val
        model.eval()
        val_loss = 0.0
        correct_num = 0
        val_accuracy = 0
        with torch.no_grad():
            for image, label in minist_val:
                image, label = image.to(device), label.to(device)
                result = model(image)
                loss = loss_F(result, label)
                val_loss += loss.mean()

                _, predicted = torch.max(result, dim=1)  # dim --> 保留的维度
                correct_num += float(predicted.eq(label).sum()) / batch_size
            val_accuracy = correct_num / len(minist_val)
            print(f"Val Epoch {epoch}: loss: {val_loss / len(minist_val)}     accuracy: {val_accuracy}\n")
            writer.add_scalar('val_loss', val_loss / len(minist_val), epoch)
            writer.add_scalars('loss',
                               {'train': total_loss / len(minist_train), 'valid': val_loss / len(minist_val)},
                               epoch)

    print("training finished")
    writer.close()

# load dataset
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
minist_train = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
minist_val = DataLoader(dataset=val_set, batch_size=64, shuffle=True)

# 设置超参数
batch_size = 64  # 批量
learning_rate = 1e-3  # 学习率
num_epochs = 10  # 训练轮数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备

# 初始化网络、损失函数和优化器、显示器
writer = SummaryWriter('runs/minist_3')  # 可视化
model = Net().to(device)  # 网络
loss_F = nn.CrossEntropyLoss()  # 损失函数
optimizer = optimizer.Adam(model.parameters(), lr=learning_rate)  # 优化器

train()
