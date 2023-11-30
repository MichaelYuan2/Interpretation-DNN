# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 00:09:46 2023

@author: MLP
"""

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
from torchvision.models import resnet34
import matplotlib.pyplot as plt

# image data path

debug1 = True 

DATAPATH = r'american_bankruptcy.csv'

logging.basicConfig(level=logging.INFO)
data = load_data(DATAPATH)

#data.to_pickle( r'data_debug' )


#data = pd.read_pickle(  r'data_debug' )

#if debug:
#    data = data.iloc[:1000]

# print(data.head())
dataset = create_dataset(data)

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load(r'models/model.pt')

import shap 
model.eval() 


def data_for_shap(loader):
    data_iter = iter(loader)
    images, _ = next(data_iter)
    return images


test_data_for_shap = data_for_shap(test_loader)
train_data_for_shap = data_for_shap(train_loader)



e = shap.DeepExplainer(model, train_data_for_shap)
shap_values = e.shap_values(test_data_for_shap, check_additivity=False)



shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_data_for_shap.numpy(), 1, -1), 1, 2)


print (shap_values)

#https://github.com/shap/shap/issues/1479
