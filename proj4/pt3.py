import torch
import torch.nn as nn
import torchvision.models as models
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision import transforms, utils

H, W = 224, 224

def create_model():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=136, bias=True)
    state_dict = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

image = io.imread('ethan.png')


image = transforms.ToPILImage()(image)
image = TF.to_grayscale(image)
image = TF.resize(image, (H, W))
image = TF.to_tensor(image).view(1, 1, H, W)

plt.imshow(image[0][0], cmap='gray')
plt.show()

with torch.no_grad():
    model = create_model()
    prediction = model(image)
    print(prediction.shape)
    prediction = prediction.view(-1, 68, 2)
    plt.imshow(image[0][0], cmap='gray')
    plt.scatter(prediction[0,:,0], prediction[0,:,1], s=10, marker='o', c='r')
    plt.show()