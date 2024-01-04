# Train-Torch

Simlified pytorch training!

## Installation

From Github:

```console
git clone https://github.com/datngu/train_torch
pip install train_pytorch
```

From PyPI:

```console
pip install train-pytorch
```


## An example on the MNIST dataset

### Load your libraries
```python

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

## import torch_trainer packages and metric functions
from train_pytorch import Trainer, binary_accuracy, multiple_class_accuracy, regression_r2

```

### Load your dataset

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

### Buid your model

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


### Let's train it!

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