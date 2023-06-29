from torch import nn

# Define the Discriminator class which inherits from nn.Module
class Discriminator(nn.Module):
    def __init__(self):
        # Call the parent class constructor
        super().__init__()
        # Define the model architecture as a sequence of layers
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Reshape the input tensor to have a size of (batch_size, 784)
        x = x.view(x.size(0), 784)
        # Pass the input through the model
        output = self.model(x)
        return output