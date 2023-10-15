---
weight: 1
title: "人工智能|机器学习开篇——从源代码开始认识机器学习"
date: 2023-10-15T11:26:13+08:00
draft: false
author: "Hang Kong"
authorLink: "https://github.com/kongh"
description: "AI|ML|从源代码开始认识机器学习"
images: []
tags: ["AI", "ML"]
categories: ["AI"]
lightgallery: true
toc:
auto: false
--- 

# 人工智能|机器学习开篇——从源代码开始认识机器学习

## 选择深度学习框架

对于初学者来说，当他们初次接触深度学习时，通常首要问题是如何选择合适的深度学习框架。截止2023年，推荐的框架有：

1. Tensorflow
2. Keras
3. PyTorch
4. MxNet
5. Chainer
6. Caffe
7. Theano
8. Deeplearning4j
9. CNTK
10. Torch

我也是个初学者，这里就不过多介绍框架的异同，若有兴趣可以点击[查看详细介绍](https://viso.ai/deep-learning/deep-learning-frameworks/)。在这篇文章里我们选择`PyTorch`作为深度学习框架。

## 使用PyTorch构建模型

### 准备数据

在机器学习中，`算法`和`数据`都是同等重要的，而获取、处理数据的过程甚至都是困难的。在本例中使用已收集且已特征化的数据来完成实例。

`PyTorch`加载数据使用两个抽象类型`torch.utils.data.DataLoader`和`torch.utils.data.Dataset`。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

`PyTorch`提供`TorchText`、`TorchVision`、`TorchAudio`三个领域模型来处理文本、图片、声音。本实例使用`TorchVision`来处理图片。

```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```

输出：

``` shell

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:12, 363426.38it/s]
  1%|          | 229376/26421880 [00:00<00:38, 683306.49it/s]
  3%|2         | 753664/26421880 [00:00<00:12, 2103021.02it/s]
  7%|7         | 1933312/26421880 [00:00<00:05, 4230832.83it/s]
 16%|#5        | 4096000/26421880 [00:00<00:02, 8795312.47it/s]
 25%|##4       | 6488064/26421880 [00:00<00:01, 10953907.49it/s]
 33%|###3      | 8781824/26421880 [00:01<00:01, 13787730.37it/s]
 43%|####2     | 11304960/26421880 [00:01<00:01, 14240885.91it/s]
 51%|#####     | 13434880/26421880 [00:01<00:00, 15828233.90it/s]
 61%|######1   | 16121856/26421880 [00:01<00:00, 15867843.04it/s]
 69%|######9   | 18284544/26421880 [00:01<00:00, 17128656.03it/s]
 79%|#######9  | 20938752/26421880 [00:01<00:00, 16653244.99it/s]
 87%|########7 | 23101440/26421880 [00:01<00:00, 17698087.30it/s]
 98%|#########7| 25821184/26421880 [00:01<00:00, 17099782.17it/s]
100%|##########| 26421880/26421880 [00:02<00:00, 13200454.99it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 328749.46it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:11, 364027.04it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 683568.60it/s]
 16%|#5        | 688128/4422102 [00:00<00:01, 1881499.36it/s]
 33%|###3      | 1474560/4422102 [00:00<00:00, 3106671.48it/s]
 84%|########4 | 3735552/4422102 [00:00<00:00, 8182214.36it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 5951236.04it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 35927249.57it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

### 创建模型

为了创建一个神经网络，可以创建一个class继承至`nn.Module`，然后在`__init__`函数中初始化神经网络的层。
```python

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```

输出：

``` shell
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

### 优化模型参数

为了训练模型，我们需要一个`损失函数`和`优化器`。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

在一次训练中，模型使用数据集进行训练，并反向传播预测误差去调整模型参数。

```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

为了测试训练的表现，我们使用`test`方法进行测试。

```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

训练过程需要进行多次迭代，可以像如下过程一样处理。

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

输出：

``` shell
Epoch 1
-------------------------------
loss: 2.303494  [   64/60000]
loss: 2.294637  [ 6464/60000]
loss: 2.277102  [12864/60000]
loss: 2.269977  [19264/60000]
loss: 2.254235  [25664/60000]
loss: 2.237146  [32064/60000]
loss: 2.231055  [38464/60000]
loss: 2.205037  [44864/60000]
loss: 2.203240  [51264/60000]
loss: 2.170889  [57664/60000]
Test Error:
 Accuracy: 53.9%, Avg loss: 2.168588

Epoch 2
-------------------------------
loss: 2.177787  [   64/60000]
loss: 2.168083  [ 6464/60000]
loss: 2.114910  [12864/60000]
loss: 2.130412  [19264/60000]
loss: 2.087473  [25664/60000]
loss: 2.039670  [32064/60000]
loss: 2.054274  [38464/60000]
loss: 1.985457  [44864/60000]
loss: 1.996023  [51264/60000]
loss: 1.917241  [57664/60000]
Test Error:
 Accuracy: 60.2%, Avg loss: 1.920374

Epoch 3
-------------------------------
loss: 1.951705  [   64/60000]
loss: 1.919516  [ 6464/60000]
loss: 1.808730  [12864/60000]
loss: 1.846550  [19264/60000]
loss: 1.740618  [25664/60000]
loss: 1.698733  [32064/60000]
loss: 1.708889  [38464/60000]
loss: 1.614436  [44864/60000]
loss: 1.646475  [51264/60000]
loss: 1.524308  [57664/60000]
Test Error:
 Accuracy: 61.4%, Avg loss: 1.547092

Epoch 4
-------------------------------
loss: 1.612695  [   64/60000]
loss: 1.570870  [ 6464/60000]
loss: 1.424730  [12864/60000]
loss: 1.489542  [19264/60000]
loss: 1.367256  [25664/60000]
loss: 1.373464  [32064/60000]
loss: 1.376744  [38464/60000]
loss: 1.304962  [44864/60000]
loss: 1.347154  [51264/60000]
loss: 1.230661  [57664/60000]
Test Error:
 Accuracy: 62.7%, Avg loss: 1.260891

Epoch 5
-------------------------------
loss: 1.337803  [   64/60000]
loss: 1.313278  [ 6464/60000]
loss: 1.151837  [12864/60000]
loss: 1.252142  [19264/60000]
loss: 1.123048  [25664/60000]
loss: 1.159531  [32064/60000]
loss: 1.175011  [38464/60000]
loss: 1.115554  [44864/60000]
loss: 1.160974  [51264/60000]
loss: 1.062730  [57664/60000]
Test Error:
 Accuracy: 64.6%, Avg loss: 1.087374

Done!
```

### 保存模型

一般通用的保存模型的方法是序列化内部的状态字典（包含模型参数）。

```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

输出：

```shell
Saved PyTorch Model State to model.pth
```

### 加载模型

为了加载一个模型需要重新创建相同的模型结构且加载状态字典。

```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
```

输出：

``` shell
<All keys matched successfully>
```

使用模型进行预测。

```python

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

```

输出：

``` shell
Predicted: "Ankle boot", Actual: "Ankle boot"
```

## 参考链接

- [PyTorch快速向导](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters)
