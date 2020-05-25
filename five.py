# https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a

# train our data set
# training
# payoff, game, create prediction of of attack
# loss function

# look at the tutorials for training data set

# no convolutions

from torch import nn
from collections import OrderedDict

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

if __name__ == "__main__":
    main()