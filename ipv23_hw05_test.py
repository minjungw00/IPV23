# region Environmental Setting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
# endregion

# region MNIST Dataset
# MNIST Dataset
mnist_train = torchvision.datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = torchvision.datasets.MNIST(root='./', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000])

# Data Loader for MNIST
mnist_train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
mnist_val_loader   = DataLoader(mnist_val, batch_size=128, shuffle=False)
mnist_test_loader  = DataLoader(mnist_test, batch_size=128, shuffle=False)
# endregion

# region CIFAR-10 Dataset
# Define the Transforms for Training Dataset
transforms_train = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define the Transforms for Testing Dataset
transforms_test = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR-10 Dataset
cifar_train = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transforms_train)
cifar_test = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transforms_test)

# Data Loader for CIFAR-10
# cifar_train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True)
# cifar_test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False)
cifar_train_loader = DataLoader(cifar_train, batch_size=128, shuffle=True, num_workers=2)
cifar_test_loader = DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=2)
# endregion

# region (Practice) Implement Each Component of CNNs
# Example of convolutional layer

# Input dimension: 1 x 3 x 32 x 32
# Convolutional layer: 32 5x5 filters with stride 2, padding 2

x = torch.randn(1, 3, 32, 32) # input: x

conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)

print('Input size:\n', x.size())
print()
print('Output size:\n', conv_layer(x).size())


# Batch Normalization

x = torch.randn(1, 3, 32, 32)

bn = nn.BatchNorm2d(num_features=3)

print('Input size:\n', x.size())
print()
print('Size of feature after BN:\n', bn(x).size()) # Please check the output size after the batch normalization whether the size of input is changed or not
# endregion

# region (Practice) Build Simple Convolutional Neural Networks
# Model: Simple Convolutional Neural Networks

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 input image channel, 32 output channels, 7x7 square convolution, 1 stride
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 32 input image channel, 64 output channels, 7x7 square convolution, 1 stride
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc = nn.Linear(64*16*16, 10)

    def forward(self, x):
        out_conv1 = self.conv_layer1(x)
        out_conv2 = self.conv_layer2(out_conv1)
        feature_1d = torch.flatten(out_conv2, 1)
        out = self.fc(feature_1d)
        return out

# Using GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = ConvNet()
model = model.to(device)


# Optimizer: Stochastic Gradient Descent Method

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define Loss function (Cross Entropy Loss here)

loss_fn = nn.CrossEntropyLoss()


# Train the model
total_step = len(mnist_train_loader)
epochs = 10
for epoch in range(epochs):
    for i, (images, labels) in enumerate(mnist_train_loader):  # mini batch for loop
        
        # Upload to gpu
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass & Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))
            

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in mnist_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of Simple CNN on MNIST test set: {} %'.format(100 * correct / total))

# endregion


# Change the following CNNs architecture

class myConvNet(nn.Module):

    def __init__(self):
        super(myConvNet, self).__init__()
        # 3 input image channel, 32 output channels, 7x7 square convolution, 1 stride
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 32 input image channel, 64 output channels, 7x7 square convolution, 1 stride
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc = nn.Linear(64*20*20, 10)

    def forward(self, x):
        out_conv1 = self.conv_layer1(x)
        out_conv2 = self.conv_layer2(out_conv1)
        feature_1d = torch.flatten(out_conv2, 1)
        out = self.fc(feature_1d)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = myConvNet()
model = model.to(device)


# Optimizer: Stochastic Gradient Descent Method
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define Loss function
loss_fn = nn.CrossEntropyLoss()


# Train the model
total_step = len(cifar_train_loader)
epochs = 5
for epoch in range(epochs):
    for i, (images, labels) in enumerate(cifar_train_loader):  # mini batch for loop
        
        # Upload to gpu
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass & Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))
            

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in cifar_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of Your CNNs on CIFAR-10 test set: {} %'.format(100 * correct / total))