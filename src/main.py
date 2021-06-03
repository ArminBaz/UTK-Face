'''
    Main Script for Gender, Age, and Ethnicity identification on the cleaned UTK Dataset.

    The dataset can be found here (https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv)

    I implemented a MultiLabelNN that performs a shared high-level feature extraction before creating a 
    low-level neural network for each classification desired (gender, age, ethnicity).
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# Custom imports
from CustomUTK import UTKDataset
from MultNN import TridentNN

def read_data():
    # Read in the dataframe
    dataFrame = pd.read_csv('../data/age_gender.gz', compression='gzip')

    # Split into training and testing
    train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2)

    # get the number of unique classes for each group
    '''
    class_nums = {'age_num':len(dataFrame['age'].unique()), 'eth_num':len(dataFrame['ethnicity'].unique()),
                  'gen_num':len(dataFrame['gender'].unique())}
    '''
    class_nums = {'age_num':1, 'eth_num':len(dataFrame['ethnicity'].unique()),
                  'gen_num':len(dataFrame['gender'].unique())}

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

    return train_loader, test_loader, class_nums

def train(trainloader, model, hyperparameters):
    # Load hyperparameters
    learning_rate = hyperparameters['learning_rate']
    num_epoch = hyperparameters['epochs']

    # Define loss functions
    age_loss = nn.MSELoss()
    gen_loss = nn.CrossEntropyLoss() # TODO : Explore using Binary Cross Entropy Loss?
    eth_loss = nn.CrossEntropyLoss()

    # Define optimizer
    opt = torch.optim.Adam(model.params(), lr=learning_rate)

    # Initialize the summaryWriter
    # writer = SummaryWriter(f'LR: {learning_rate}')

    # Train the model
    for epoch in range(num_epoch):
        # Construct tqdm loop to keep track of training
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        correct, total = 0,0
        # Loop through dataLoader
        for _, (X,y) in loop:
            # unpack y to get true age, eth, and gen values
            age = y[:,0]
            gen = y[:,1]
            eth = y[:,2]

            pred = model(X)          # Forward pass
            loss = age_loss(pred[0],age) + gen_loss(pred[1],gen) + eth_loss(pred[2],eth)   # Loss calculation

            # Backpropagation
            opt.zero_grad()          # Zero the gradient
            loss.backward()          # Calculate updates
            
            # Gradient Descent
            opt.step()               # Apply updates

            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
            loop.set_postfix(loss = loss.item())



def test(testloader):
    pass

def main():
    # Define the list of hyperparameters
    hyperparams = {'learning_rate':0.001, 'epochs':50, }
    
    # Read in the data and store in train and test dataloaders
    train_loader, test_loader, class_nums = read_data()
    
    # Initialize the TridentNN model
    tridentNN = TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])

    # Train the model
    train(train_loader, tridentNN, hyperparams)

if __name__ == '__main__':
    main()