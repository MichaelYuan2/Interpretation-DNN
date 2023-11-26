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
from torchvision.models import resnet50

# image data path

DATAPATH = r'american_bankruptcy.csv'

logging.basicConfig(level=logging.INFO)
data = load_data(DATAPATH)
dataset = create_dataset(data)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load(r'models/model.pth')
except:
    resnet50 = resnet50(pretrained=False)
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_features, 2)
    resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = resnet50.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), torch.tensor(target).to(device) 
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%100 == 0: #Print result every 100 batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), torch.tensor(target).to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() 
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_accuracy = test(model, device, test_loader)

torch.save(model, r'models/model.pt')