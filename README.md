# Train-Pytorch

Simlified pytorch training!

The package provide:
- A basic `Trainer` class to facilidate pytorch model training.
- Some functions to compute common accuracy metrics including:
    - `binary_accuracy`
    - `multiple_class_accuracy`
    - `regression_r2`


You can also define your own function to input into the `Trainer` class as long as your function can:
- take 2 inputs: `logits` and `labels`
- perform all computation on: `torch.tensor` 
- return a python value by: `value.item()`

An example of our provided `binary_accuracy` function is:

```python

def binary_accuracy(logits, labels, cutoff=0):
    """
    Compute binary classification accuracy score.
        return accuracy value

    Args:
        logits: logits - outputs of the model
        labels: true labels of data
        cutoff: default is 0 - model outputs logits
                can be set to 1 - if model outputs probabilities
    """
    logits, labels = logits.cpu(), labels.cpu()
    predicts = (logits > cutoff).float()
    acc = (predicts == labels).float().mean()
    return acc.item()

```





## 1. Installation

From Github:

```console
git clone https://github.com/datngu/train_pytorch
cd train_pytorch
pip install .
```

From PyPI:

```console
pip install train-pytorch
```


## 1. Example on the MNIST dataset with multiple_class_accuracy

### 1.1 Load your libraries
```python

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

## import train_pytorch packages and metric functions
from train_pytorch import Trainer, binary_accuracy, multiple_class_accuracy, regression_r2

```

### 1.2 Load your dataset

```python

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

```

### 1.2 Buid your model

```python

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


```


### 1.3 Let's train it!

```python

model = CNNModel()

## GPU: optional
#device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
#model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


trainer = Trainer(model, criterion, optimizer, multiple_class_accuracy, num_epochs = 10, early_stoper = 5)

trainer.fit(train_loader, train_loader, './output_dir')

```


## 2. Example on the sklearn breast_cancer dataset with binary_accuracy

### 2.1 Load your libraries
```python

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from torch.utils.data import Dataset, DataLoader

## import train_pytorch packages and metric functions
from train_pytorch import Trainer, binary_accuracy, multiple_class_accuracy, regression_r2

```


### 2.2 Load your dataset

```python

data = load_breast_cancer()
x = data['data']
y = data['target']

sc = StandardScaler()
x = sc.fit_transform(x)

## create dataset class
class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
 
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length
      
# a bit lazy to slipt train and test data, but it is okey for tutorial :D      
train_data = dataset(x,y)
val_data = dataset(x,y)

train_loader = DataLoader(train_data,batch_size=64,shuffle=False)
val_loader = DataLoader(val_data,batch_size=64,shuffle=False)


```


### 2.2 Buid your model

```python

class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(32,64)
    self.fc3 = nn.Linear(64,1)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x


```


### 2.3 Let's train it!

```python

model = Net(input_shape=x.shape[1])

## GPU: optional
#device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
#model.to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()


trainer = Trainer(model, loss_fn, optimizer, binary_accuracy, num_epochs=10)
trainer.fit(train_loader, val_loader, './output_dir')

```