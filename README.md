# UTK-Face Project
This is a project based on the UTK Face dataset, luckily someone was nice enough to clean up the dataset
and make it publicly available here: https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv
<br>
<br>
This repo contains a multi-output CNN that is capable of predicting the age, gender, and ethnicity of a person from a photo.

## Dependencies
* python - 3.7.4
* pandas - 1.0.5
* pytorch - 1.4.0
* torchvision - 0.5.0
* tensorboard - 2.4.1
* tqdm - 4.47.0

## Usage
To run the script navigate to `/src` and run `main.py` with the following command

``` sh
    python main.py --num_epochs=e --learning_rate=lr --pre-trained=pt
```
The argument usage should be self explanatory, further information can be found in the main.py file.

## Data
The data is stored in a gzip format to make it easier to download and work with

## Src
Within this folder you will find two custom modules
1. The first custom module is for a custom PyTorch Dataset class to work with the UTK dataset
2. The second custom module is where I define my trident neural network

## Trident Neural Network
The trident neural network is named such because it takes the shape of an upside down trident.
We are trying to predict three different things with each image age, gender, and ethnicity.


The steps for the TNN are as follows:
1. I run the image through the high-level feature extraction layers that closely mimic the architecture of the VGG network.
2. The high-level features are sent through three seperate branches (hence the trident).
3. Within each branch low-level features are extracted followed by fully connected layers for classification.

As you can assume the three different branches are for the three predictions.

## Test Results
I got these test results after training the network so that the training accuracy for each task hit 90%. You can see how I did that in the horribly unorganized colab notebook in `/colab/TrainTridentMain.ipynb`.

| Task        | Accuracy      |
| ----------- | -----------   |
| Age         | 45.86%        |
| Gender      | 89.01%        |
| Ethnicity   | 76.46%        |

As you can see the accuracy is not that great, especially for the age prediction task. My hypothesis is that predicting age is actually a very difficult task because there is not only variation between how people age from person to person, but different genders and ethnicities show age at different rates too. Therefore, we have both inter and intra variance when it comes to age. <br> <br>

I might revisit this problem in the future by first trying to feedback the gender and ethnicity predictions as one hot vectors into the age prediction branch of the neural network. Or, it might be as simple as using a more powerful method that involves using a better network (ResNet/DenseNet) and/or the use of transfer learning.
