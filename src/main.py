'''
    Main Script for Gender, Age, and Ethnicity identification on the cleaned UTK Dataset.

    The dataset can be found here (https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv)

    I implemented a MultiLabelNN that performs a shared high-level feature extraction before creating a 
    low-level neural network for each classification desired (gender, age, ethnicity).
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
# Custom imports
from CustomUTK import UTKDataset
from MultNN import TridentNN

'''
    Add arguments for argparse

    Arguments:
     - epochs : (int) Number of epochs to train
     - lr : (float) Learning rate for the model
     - pre-trained : (bool) whether or not to load the pre-trained model
'''
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--num_epochs', default=20, type=int, help='(int) number of epochs to train the model')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='(float) learning rate of optimizer')
parser.add_argument('-pt', '--pre-trained', default=True, type=bool, help='(bool) whether or not to load the pre-trained model')


'''
    Function to read in the data
    Inputs: None

    Outputs:
     - train_loader : Custom PyTorch DataLoader for training data from UTK Face Dataset
     - test_loader : Custom PyTorch DataLoader for testing UTK Face Dataset
     - class_nums : Dictionary that stores the number of unique variables for each class (used in NN)
'''
def read_data():
    # Read in the dataframe
    dataFrame = pd.read_csv('../data/age_gender.gz', compression='gzip')

    # Construct age bins
    age_bins = [0,10,15,20,25,30,40,50,60,120]
    age_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    dataFrame['bins'] = pd.cut(dataFrame.age, bins=age_bins, labels=age_labels)

    # Split into training and testing
    train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2)

    # get the number of unique classes for each group
    class_nums = {'age_num':len(dataFrame['bins'].unique()), 'eth_num':len(dataFrame['ethnicity'].unique()),
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


'''
   Function to train the model

   Inputs:
     - trainloader : PyTorch DataLoader for training data
     - model : NeuralNetwork model to train
     - opt : Optimizer to train with
     - num_epoch : How many epochs to train for

    Outputs: Nothing
'''
def train(trainloader, model, opt, num_epoch):
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Define loss functions
    age_loss = nn.CrossEntropyLoss()
    gen_loss = nn.CrossEntropyLoss()
    eth_loss = nn.CrossEntropyLoss()


    # Train the model
    for epoch in range(num_epoch):
        # Construct tqdm loop to keep track of training
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        gen_correct, eth_correct, age_correct,total = 0,0,0,0    # capital l on age to not get confused with loss function
        # Loop through dataLoader
        for _, (X,y) in loop:
            # Unpack y to get true age, eth, and gen values
            # Have to do some special changes to age label to make it compatible with NN output and Loss function
            age, gen, eth = y[:,0].to(device), y[:,1].to(device), y[:,2].to(device)
            X = X.to(device)

            pred = model(X)          # Forward pass
            loss = age_loss(pred[0],age) + gen_loss(pred[1],gen) + eth_loss(pred[2],eth)   # Loss calculation

            # Backpropagation
            opt.zero_grad()          # Zero the gradient
            loss.backward()          # Calculate updates
            
            # Gradient Descent
            opt.step()               # Apply updates

            # Update num correct and total
            age_correct += (pred[1].argmax(1) == age).type(torch.float).sum().item()
            gen_correct += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
            eth_correct += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

            total += len(y)

            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
            loop.set_postfix(loss = loss.item())

    # Update epoch accuracy
    gen_acc, eth_acc, age_acc = gen_correct/total, eth_correct/total, age_correct/total

    # print out accuracy and loss for epoch
    print(f'Epoch : {epoch+1}/{num_epoch},    Age Accuracy : {age_acc*100},    Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n')


'''
    Function to test the trained model

    Inputs:
      - testloader : PyTorch DataLoader containing the test dataset
      - modle : Trained NeuralNetwork
    
    Outputs:
      - Prints out test accuracy for gender and ethnicity and loss for age
'''
def test(testloader, model):
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    size = len(testloader.dataset)
    # put the moel in evaluation mode so we aren't storing anything in the graph
    model.eval()

    age_acc, gen_acc, eth_acc = 0, 0, 0  # capital L on age to not get confused with loss function

    with torch.no_grad():
        for X, y in testloader:
            age, gen, eth = y[:,0].to(device), y[:,1].to(device), y[:,2].to(device)
            X = X.to(device)
            pred = model(X)

            age_acc += (pred[0].argmax(1) == age).type(torch.float).sum().item()
            gen_acc += (pred[1].argmax(1) == gen).type(torch.float).sum().item()
            eth_acc += (pred[2].argmax(1) == eth).type(torch.float).sum().item()

    age_acc /= size
    gen_acc /= size
    eth_acc /= size

    print(f"Age Accuracy : {age_acc*100}%,     Gender Accuracy : {gen_acc*100},    Ethnicity Accuracy : {eth_acc*100}\n")


'''
    Main function that stiches everything together
'''
def main():
    # Configure the device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse the arguments
    args = parser.parse_args()

    # Read in the data and store in train and test dataloaders
    train_loader, test_loader, class_nums = read_data()

    # Load the model and optimizer
    tridentNN = TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])

    # Define optimizer
    opt = torch.optim.Adam(tridentNN.parameters(), lr=args.lr)

    # If we are training from scratch
    if args.pt == False:
        # Train the model
        train(train_loader, tridentNN, opt, args.e)
        print('Finished training, running the testing script...\n \n')
        test(test_loader, tridentNN)
    else:
        # Load and test the pre-trained model
        checkpoint = torch.load('/checkpoints/tridentNN_epoch20.pth.tar')
        tridentNN.load_state_dict(checkpoint['state_dict'])
        test(test_loader, tridentNN)

if __name__ == '__main__':
    main()