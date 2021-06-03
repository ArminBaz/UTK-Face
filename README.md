# UTK-Face Project
This is a project based on the UTK Face dataset, luckily someone was nice enough to clean up the dataset
and make it publicly available here: https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv

## Data
The data is stored in a gzip format to make it easier to download and work with

## Src
Within this folder you will find two custom modules
1. The first custom module is for a custom PyTorch Dataset class to work with the UTK dataset
2. The second custom module is where I define my trident neural network

## Trident Neural Network
The trident neural network is named such because it takes the shape of an upside down trident.
We are trying to predict three different things with each image age, gender, and ethnicity. 
I first run the image through a high-level feature extractor that mimics the architecture of the famed VGG network. After the high-level features are extracted, they are sent through three seperate branches for low-level feature extraction and classification.

As you can assume the three different branches are for the three predictions.
