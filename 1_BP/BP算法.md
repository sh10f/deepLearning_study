**MINIST手写体识别 ( Pytorch实现 )**

* 所需知识：**BP算法、线性模型、Pytorch训练框架**



# BP算法

* 在 <u>前馈神经网络</u> 中常用BP算法来实现 **权重优化**。

* 训练流程

  * 前向传播 ——> 损失函数
    * 将 *mini-batch* 的样本传入 *model* 进行训练，得到  $\hat{y}$ 。将其与 *label* 进行对比计算 *loss*.
  * BP——> 梯度下降
    * 对 *loss* 进行求偏导， 并通过 *optimizer* 对权重参数进行调整

* 缺点

  * 反向传播过程中，由于不断求偏导，导致 **输入层与第1层之间的权重梯度公式中含有多个偏导$f'(x)$**, 导致 **梯度消失**。

  

# 线性模型

* 层与层之间可以采用 **全连接** 的方式，，即 **一对全部** 的关系(上层神经元对下层)； 层内无连接



# 训练

## 训练产生问题

* 过拟合
  1. **过度拟合训练集**。训练次数越多，验证集上效果先变好，**后又变差**
  2. 解决方法
     - 数据增强
     - 降低模型复杂度
       1. Dropout
       2. 惩罚性损失函数

## Pytorch 训练流程

1. 导入基本包

  ```python
  import os
  import numpy as np
  import torch
  import torch.nn as nn
  from torch.utils.data import Dataset, DataLoader
  from torchvision.transforms import transforms
  from torchvision import datasets
  import torch.optim as optimizer
  from tensorboardX import SummaryWriter
  ```

  

2. 加载数据集 —— train 和 val

  * 开源数据集 ———— 针对不同的领域使用不同包自带的数据集（eg: cv---torchvision.datasets)

    <img src=".\\imgs_log\\领域对应库.png" width=20% >

    ```python
    # load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    minist_train = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    minist_val = DataLoader(dataset=val_set, batch_size=64, shuffle=True)
    ```

    

  * 自定义数据集

    ```python
    from torch.utils.data import Dataset
    
    class Dataset(Generic[T_co]):
        r"""An abstract class representing a :class:`Dataset`.
    
        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~torch.utils.data.DataLoader`. Subclasses could also
        optionally implement :meth:`__getitems__`, for speedup batched samples
        loading. This method accepts list of indices of samples of batch and returns
        list of samples.
    
        .. note::
          :class:`~torch.utils.data.DataLoader` by default constructs an index
          sampler that yields integral indices.  To make it work with a map-style
          dataset with non-integral indices/keys, a custom sampler must be provided.
        """
    
        def __getitem__(self, index) -> T_co:
            raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")
    ```

  

3. 建立 **训练模型**

  ```python
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
  ```

  

4. 设置训练参数 --- **超参数** 和 **训练所需模块**

  ```python
  # 设置超参数
  batch_size = 64  # 批量
  learning_rate = 1e-3  # 学习率
  num_epochs = 10  # 训练轮数
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备
  
  # 初始化网络、损失函数和优化器、显示器
  writer = SummaryWriter('runs/minist_2')  # 可视化
  model = Net().to(device)  # 网络
  loss_F = nn.CrossEntropyLoss()  # 损失函数
  optimizer = optimizer.Adam(model.parameters(), lr=learning_rate)  # 优化器
  ```

  

5. **训练**

  ```python
  def train():
      """a fixed training flow
      
      1.训练过程
          1.1 在 optimizer.zero_grad() 梯度 清零 的情况下。
          1.2 每个epoch里，按照 batch_size 遍历 dataloader 计算损失值。 
          1.3 通过 minibatch 的样本所计算的损失值，对模型 权重 进行梯度下降修改。
          
          1.4 每个 epoch 结束后，计算 总体的损失
      
      2. 验证过程
          2.1 需在评估模式下进行，并设置 无梯度模式：
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
  
  train()
  ```





# MINIST结果

<img src= ".\imgs_log\result.png" width=40% style="float:left" >  <img src= ".\imgs_log\loss.png" width=50% style="float:right" >

































# Conclusion

* 对于线性模型的实现基本掌握
* 对于 **超参数设置** 和 **训练工具选择（损失函数，优化器等）**有了更深的了解
  * lr 过大会导致 loss 上升。 分为 static 和 dynamic
  * epoch 和 batch_size 的选择应当适当
  * 损失函数有 多种选择（ie. Sigmoid，Relu，leakyRelu， tanh）
  * 优化器的作用（eg：momentum）对权重参数进行调整



**end**







