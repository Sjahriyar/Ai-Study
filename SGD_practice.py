import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

# Download training data from open datasets.
batch_size = 64

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# This code defines a neural network with three fully connected (Linear) layers and two activation functions (ReLU). 
# The layers are applied sequentially, meaning the output of one layer becomes the input to the next.
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    # same as `.view(-1, 28*28)` it flattens the input tensor to 1D tensor
    self.flatten = nn.Flatten()
    # Randomly initialize weights
    # nn.Sequential is a container for stacking layers in sequence.
    self.linear_relu_stack = nn.Sequential(
      # 28x28 means how many features our input has? 28*28 = 784
      # In our 28x28 image, we have 784 features
      # 512: The number of neurons (or units) in this layer. Each neuron computes a weighted sum of the inputs and applies a bias.
      # For more information, see the optional reads: "Number of neurons (or units) in the hidden layers" section.
      # Why using Linear layer? READ: optinal_read.ipynb#linear-model
      # The weights are part of the nn.Linear layers in your model. 
      # Each nn.Linear layer creates its own set of weights and biases when it is initialized.
      nn.Linear(28*28, 512),

      # ReLU is the activation function. It introduces non-linearity into the model.
      # The ReLU activation function is applied element-wise to the output of the first layer, setting all negative values to 0, and leaving all other values unchanged.
      # ReLU(x)=max(0,x)
      nn.ReLU(),

      # The second fully connected layer accepts the output of the first layer as input, applies a bias, and outputs a 512-dimensional tensor.
      nn.Linear(512, 512),
      nn.ReLU(),

      # The final fully connected layer receives the 512-dimensional tensor from the second layer and outputs a 10-dimensional tensor.
      # 10 because our dataset has 10 classes.
      # FashionMNIST has 10 classes (e.g. 0: T-shirt, 1: Trouser, 2: Pullover) so we have 10 output channels.
      nn.Linear(512, 10),
    )

  # The forward function defines how input data (x) flows through the neural network to produce an output. 
  # It specifies the sequence of operations that transform the input into predictions (logits). 
  # This function is called automatically when you pass data through the model (e.g., model(x)).
  def forward(self, x):
    x = self.flatten(x)                 # Step 1: Flatten the input
    logits = self.linear_relu_stack(x)  # Step 2: Pass through the layers
    return logits                       # Step 3: Return the output

def train(dataloader, model, loss, optimizer):
  size = len(dataloader.dataset)
  model.train()

  for batch, (data, label) in enumerate(dataloader):
    data, label = data.to(device), label.to(device)

    # Compute prediction error
    pred = model(data)
    loss = loss_fn(pred, label)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(data)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = NeuralNetwork().to(device)

# To train a model we need a loss function and an optimizer

loss_fn = nn.CrossEntropyLoss()
# 1e-3 means that the optimizer will adjust the model parameters by the factor of 1e-3 (0.001) times the gradient.
# This is a hyperparameter that we can tune.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

torch.save(model.state_dict(), "fashion.pth")
print("Saved PyTorch Model State to fashion.pth")
