import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets
import torch.optim as optimizer
from tensorboardX import SummaryWriter


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()

        self.__initialize_weights()

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)

        return x


batch_size = 32

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, ), (0.5, ))])
trainset = datasets.MNIST('../DataSets', train=True, download=True, transform=transforms)
validset = datasets.MNIST('../DataSets', train=False, download=True, transform=transforms)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
lr = 0.001

# Early Stopping Parameters
early_stopping_count = 0
best_val_loss = 1e10
patience = 10


writer = SummaryWriter(r'runs\\minist_test')
model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optim = optimizer.Adam(model.parameters(), lr=lr)

dummy_input, _ = next(iter(trainloader))
writer.add_graph(model, dummy_input.to(device))

for epoch in range(epochs):
    model.train()  # 确保模型在训练模式
    total_loss = 0
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optim.zero_grad()  # 优化器梯度清零 在计算前

        output = model(data)
        loss = loss_fn(output, target)

        loss.backward()
        optim.step()
        total_loss += loss.item()
    print("Train Epoch {} Loss: {}".format(epoch, total_loss / len(trainloader)))
    writer.add_scalar('train_loss', total_loss / len(trainloader), epoch)

    model.eval()
    with torch.no_grad():
        correct = 0
        val_loss = 0
        val_accuracy = 0
        for data, target in validloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target)

            _, predicted = torch.max(output, 1)  # 返回 最大值 和 index
            correct += predicted.eq(target).sum().item()

        avg_val_loss = val_loss / len(validloader)
        val_accuracy = correct / len(validloader) / batch_size
        print("Validation Epoch {} Loss: {}, Accuracy: {}\n".format(epoch, avg_val_loss, val_accuracy))
        writer.add_scalars('val',
                           {"val_loss": val_loss, "val_accuracy": val_accuracy}, epoch)

        writer.add_scalars("train & loss",
                           {"train_loss": total_loss / len(trainloader), "val_loss": avg_val_loss},
                           epoch)

        # add Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_count = 0
            torch.save(model.state_dict(), 'best_model_test.pth')
        else:
            early_stopping_count += 1
            if early_stopping_count >= patience:
                print("Early stopping")
                break


print("training finished")
writer.close()
