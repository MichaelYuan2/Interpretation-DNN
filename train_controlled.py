import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from create_dataset import *
from tqdm import tqdm
from PIL import Image
import logging
import torch.nn.functional as F
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# image data path

DATAPATH = r'american_bankruptcy.csv'
# IDXPATHS = [r'index/SHAP_label0_Top10.npy', r'index/SHAP_label1_Top10.npy']
IDXPATHS = [r'index/GradCAM_label0_Top10.npy', r'index/GradCAM_label1_Top10.npy']

logging.basicConfig(level=logging.INFO)
data = load_data(DATAPATH)

# Load top 10 ratios's indexes, for class 0 and class 1
idxes = []
for idx_path in IDXPATHS:
    idxes.append(np.load(idx_path))
print(idxes[0])

dataset = create_dataset_controlled(data, idxes)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layer 1
        self.fc1 = nn.Linear(64 * 16 * 16, 64)  # input image size is 64x64
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.tensor(target).type(torch.LongTensor).to(device) 
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Scale loss by batch size
        train_loss += loss.item() * data.size(0)  
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]  
        # update correct prediction 
        correct += pred.eq(target.view_as(pred)).sum().item()

        if(batch_idx+1)%100 == 0: #Print result every 100 batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    train_loss /= len(train_loader.dataset)
    train_accuracy = correct / len(train_loader.dataset)
    return train_loss, train_accuracy

            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), torch.tensor(target).type(torch.LongTensor).to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() 
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (test_loss, correct / len(test_loader.dataset))

def plot_loss(epochs, train_losses, test_losses, fp = 'plot'):
    plt.plot(range(epochs), train_losses, label='Train')
    plt.plot(range(epochs), test_losses, label='Test')
    plt.grid()
    plt.title('Loss Curves_controlled')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fp, "Loss_Curves_controlled_GradCAM1.png"))
    plt.clf()


def plot_accuracy(epochs, train_acc, test_acc, fp = 'plot'):
    plt.plot(range(epochs), train_acc, label='Train')
    plt.plot(range(epochs), test_acc, label='Test')
    plt.grid()
    plt.title('Accuracy Curves_controlled')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(fp, "Accuracy_Curves_controlled_GradCAM1.png"))
    plt.clf()


train_losses = []
test_losses = []
train_acc = []
test_acc = []
for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_accuracy = test(model, device, test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)

try:
    os.mkdir(r"models")
except:
   pass

model = model.to('cpu')
torch.save(model, r'models/model_GradCAM1_controlled.pt')

try:
    os.mkdir(r"plot")
except:
    pass
plot_loss(epochs, train_losses, test_losses)
plot_accuracy(epochs, train_acc, test_acc)