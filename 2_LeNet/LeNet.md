# LeNet

## 模型架构

* 两次 <Convolution， MaxPooling>
* 三次 \<Full Connection>

## LeNet注意

* DropOut层 应该在 **激活函数后面**
  * 可以把 $\sigma(WX + b)$ 看做一个神经元整体 
* self.__initialize_weights() 双下划线表示 **私有函数**
* 在 权重初始化 时，若要进行细粒度的初始化，就必须每层都实例化。难以做到每层的复用
  * forward() 调用实例化的对象时， 一定要注意 **带权重的层 和不带权重的层**
  * 带权重的层难以复用，因为复用时的权重参数会修改，影响上次调用该层
* **全连接层前必须将 data 展开成二维的数据**

```python
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
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x
```





# MINIST训练

## 第二轮训练

* 

<img src=".\imgs_log\model_structure.png" width="35%" style="float:left" ><img src=".\imgs_log\lenet_result_2.png" width=40% style="float:right" >













































## 第三次训练（epoch= 50）

* 出现 **过拟合现象**

<img src=".\imgs_log\loss_3st_full.png" width=75% >

* 解决方法 ———— Early Stopping

  * ```python
    early_stop_count = 0
    best_val_loss = float('inf')
    patience = 10  # 提前停止的耐心轮数
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in trainloader:
            xxxxx
            
            
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for data, target in validloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_fn(output, target)
    
            val_loss / = len(validloader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_count = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                early_stop_count += 1
                if early_stop_count >= patience:
                    print("Early stopping")
                    break
    
    
    ```

    

### Early Stopping

<img src=".\imgs_log\loss_earlystop.svg"  width=70%>





# 股票价格走势预测

