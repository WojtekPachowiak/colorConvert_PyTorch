from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from utils import *

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


################################################################################


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(data, model, loss_fn, optimizer):
    size = len(data.dataset)
    model.train()
    for batch_i, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_i % 1000 == 0 or batch_i == size - 1:
            loss, current = loss.item(), (batch_i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(data, model, loss_fn):
    num_batches = len(data)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: {test_loss:>8f} \n")


################################################################################


def prepare_data(img_path):
    x = load_img(img_path)
    y = convert_rgb_to_hsv(x)

    tensor_x = torch.Tensor(x).reshape(-1, 3)
    tensor_y = torch.Tensor(y).reshape(-1, 3)

    # same size on input and output
    assert tensor_x.shape == tensor_y.shape

    train_ratio = 0.8
    train_size = int(train_ratio * len(tensor_x))
    test_size = len(tensor_x) - train_size
    train_dataset = TensorDataset(tensor_x[:train_size], tensor_y[:train_size])
    test_dataset = TensorDataset(tensor_x[train_size:], tensor_y[train_size:])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader


################################################################################


if __name__ == "__main__":
    train_dataloader, test_dataloader = prepare_data("rgb_square.png")

    model = NeuralNetwork().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train and test
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    # save model (with current date and time as name)
    name = f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
    torch.save(model.state_dict(), name)
    print(f"Saved PyTorch Model State to '{name}'")
