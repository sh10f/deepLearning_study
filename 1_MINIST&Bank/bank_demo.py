import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
import torch.optim as optimizer
from tensorboardX import SummaryWriter


class CustomDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (string): Directory with all .csv file.
            train (bool, optional): if True, select training set. Defaults to True
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []

        if self.train:
            self.datapath = os.path.join(self.root, 'select-data.csv')
        else:
            self.datapath = os.path.join(self.root, 'scalar-test.csv')

        self.load_data()

    def load_data(self):
        all_data = pd.read_csv(self.datapath)
        self.data = all_data.iloc[:, 1:-1:1]
        self.labels = all_data["Exited"].values.tolist()

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # transform ---- Convert a PIL Image or ndarray to tensor and scale the values accordingly.
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        data = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return data, label


class MyDataset(Dataset):
    """适用于如此 树结构 的数据集

    data/
        train/
            class1/
                img1.jpg
                img2.jpg
                ...
            class2/
                img1.jpg
                img2.jpg
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # Return a list containing the names of the files in the directory.
        self.img_paths = []
        self.labels = []

        # 加载 数据集 中的 样本地址集 和 label集
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.img_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # if self.transform:
        # image = self.transform(image)
        #
        # return image, label


# forwar() <=> 神经网络模块
class BankNet(nn.Module):
    """An Linear Model inherits from nn.Module

    隐藏层相当于3层：全连接 + 全连接
    全连接层的激活函数为 Sigmoid
    同时还有 Dropout
    """

    def __init__(self):
        super(BankNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(10, 256),
            nn.Sigmoid(),
            nn.Dropout(0.5),

            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Dropout(0.5),

            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.sequential(x)
        return x

    def _initialize_weights(self):
        for m in self.sequential:
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def trainning_now(train_set, val_set):
    """a fixed training flow

    1.训练过程
        1.1 在 optimizer.zero_grad() 梯度初始化的情况下。
        1.2 每个epoch里，按照 batch_size 遍历 dataloader 计算损失值。
        1.3 通过 minibatch 的样本所计算的损失值，对模型 权重修改。_set

        1.4 每个 epoch 结束后，计算 总体的损失

    2. 验证过程
        2.1 需在评估模式下进行，并设置无梯度：
            model.eval()
            with torch.no_grad():
                xxx

        2.2 每个 epoch 结束后，都可以验证
        2.3 验证就是将 val_set 的样本代入 model() 进行 前向传播。将得到的 result 与 label 进行对比

    """
    dummy_input, _ = next(iter(train_set))
    writer.add_graph(model, dummy_input)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for image, label in train_set:
            # RuntimeError: Expected floating point type for target with class probabilities, got Long
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = loss_F(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.mean()
        print(f"Train Epoch {epoch}: loss: {total_loss / len(train_set)}")
        writer.add_scalar('train_loss', total_loss / len(train_set), epoch)

        # val
        model.eval()
        val_loss = 0.0
        correct_num = 0
        val_accuracy = 0
        with torch.no_grad():
            for image, label in val_set:
                image, label = image.to(device), label.to(device)
                result = model(image)
                loss = loss_F(result, label)
                val_loss += loss.mean()

                _, predicted = torch.max(result, dim=1)  # dim --> 保留的维度
                correct_num += float(predicted.eq(label).sum()) / batch_size
            val_accuracy = correct_num / len(val_set)
            print(f"Val Epoch {epoch}: loss: {val_loss / len(val_set)}     accuracy: {val_accuracy}\n")
            writer.add_scalar('val_loss', val_loss / len(val_set), epoch)
            writer.add_scalars('loss',
                               {'train': total_loss / len(train_set), 'valid': val_loss / len(val_set)},
                               epoch)

    print("training finished")
    writer.close()


if __name__ == '__main__':

    # load Custom dataset
    path = r'.\\data\\BANK'

    train_set = CustomDataset(root=path, train=True)
    val_set = CustomDataset(root=path, train=False)

    bank_train = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    bank_val = DataLoader(dataset=val_set, batch_size=64, shuffle=True)

    # 设置超参数
    batch_size = 64  # 批量
    learning_rate = 1e-4  # 学习率
    num_epochs = 50  # 训练轮数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备

    # 初始化网络、损失函数和优化器、显示器
    writer = SummaryWriter('runs/bank_5')  # 可视化
    model = BankNet().to(device)  # 网络
    loss_F = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optimizer.Adam(model.parameters(), lr=learning_rate)  # 优化器

    trainning_now(bank_train, bank_val)
