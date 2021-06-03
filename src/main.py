'''
    Main Script for Gender, Age, and Ethnicity identification on the cleaned UTK Dataset.

    The dataset can be found here (https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv)

    I implemented a MultiLabelNN that performs a shared high-level feature extraction before creating a 
    low-level neural network for each classification desired (gender, age, ethnicity).
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
# Custom imports
from CustomUTK import UTKDataset
from MultNN import TridentNN

# Read in the dataframe
dataFrame = pd.read_csv('../data/age_gender.gz', compression='gzip')

# Split into training and testing
train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2)

# get the number of unique classes for each group
age_num = len(dataFrame['age'].unique())
eth_num = len(dataFrame['ethnicity'].unique())
gen_num = len(dataFrame['gender'].unique())

# Define train and test transforms
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49,), (0.23,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49,), (0.23,))
])

# Construct the custom pytorch datasets
train_set = UTKDataset(train_dataFrame, transform=train_transform)
test_set = UTKDataset(test_dataFrame, transform=test_transform)

# Load the datasets into dataloaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# Sanity Check
for X, y in train_loader:
    print(f'Shape of training X: {X.shape}')
    print(f'Shape of y: {y.shape}')
    break