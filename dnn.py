# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a

# train our data set
# training
# payoff, game, create prediction of of attack
# loss function

# look at the tutorials for training data set

# review of literature
# thoughts of deep learning
# plot some graphs, prediction accuracy
# different results for each game seperately
# show results
# what if you change number of nodes and layers for hidden layers
# train in different game, train single neural net
# wrap up


# no convolutions

import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import math

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation 𝑥𝐖+𝑏xW+b
        self.hidden = nn.Linear(784, 256) # (inputs, outputs) weights
        
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid() # sigmoid activation
        self.softmax = nn.Softmax(dim=1) # softmax across columns
        self.loss = nn.L1Loss()
        
    def forward(self, x): # x is input tensor
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        x = self.L1Loss(x)
        
        return x

def main():
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    model = Network()
    # Build a feed-forward network
    '''
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.Softmax(dim=1))
    '''
    model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('output', nn.Linear(hidden_sizes[1], output_size)),
        ('softmax', nn.Softmax(dim=1))]))
    print(model[0])
    print(model.fc1)

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

    # Download and load the training data\
    trainset = datasets.MNIST('~/Users/alyssahuque/Downloads/Data/24Target-9Resource/Group1/Set1/Game_1', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = nn.Sequential(nn.Linear(784, 128),nn.ReLU(),nn.Linear(128, 64),nn.ReLU(),nn.Linear(64, 10),nn.LogSoftmax(dim=1))
    # Define the loss
    criterion = nn.NLLLoss()
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    xaxis = []
    for e in range(5):
    	running_loss = 0
    	for data, labels in trainloader:
    		# Flatten MNIST data into a 784 long vector
    		data = data.view(data.shape[0], -1)

    		# Training pass
    		optimizer.zero_grad()

    		output = model(data)
    		loss = criterion(output, labels)
    		loss.backward()
    		optimizer.step()

    		running_loss += loss.item()
    	else:
    		print(f"Training loss: {running_loss/len(trainloader)}")
    		xaxis.append(running_loss/len(trainloader))

    	
    # x axis values 
    x = xaxis
    # corresponding y axis values 
    y = [1,2,3,4,5]

    # plotting the points  
    plt.plot(x, y) 
    # naming the x axis 
    plt.xlabel('training loss') 
    # naming the y axis 
    plt.ylabel('each backward pass') 

    # giving a title to my graph 
    plt.title('Training DNN') 

    # function to show the plot 
    plt.show()

if __name__ == "__main__":
    main()