import torch
from utils import *
from main import NeuralNetwork
import numpy

# load the saved model
model = NeuralNetwork()
saved_model_path = 'model_2023-04-30_16-00-08.pth'
model.load_state_dict(torch.load(saved_model_path))

print(model)
# set the model to evaluation mode
model.eval()


################################################################################


img_path = "rgb_square.png"
new_data = load_img(img_path)
new_data = torch.Tensor(new_data).reshape(-1, 3)
print(new_data.shape)
# new_data = torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.3, 0.0, 0.3]])
# print(new_data)

# make predictions on the new data
with torch.no_grad():
    pred = model(new_data)
    print(pred)
    print(pred.shape)

    plot("rgb_square.png", pred.reshape(1024, 1024, 3).numpy())

