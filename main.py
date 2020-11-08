from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim import Adam
import cv2

#import custom classes
from dataset import FaceDataset, create_dataloader, test_dataloader
from pt2_new import pt_2

root_dir = './imm_face_db/'

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 12->20->28 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 26, 5)
        self.conv3 = nn.Conv2d(26, 30, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(720, 240)  # 12*16 from image dimension
        self.fc2 = nn.Linear(240, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def pt_1():
    epochs = 25
    training_width = 80
    training_height = 60

    face_dataset = FaceDataset(1, 32, root_dir, training_width, training_height)
    training = create_dataloader(face_dataset, 5)

    validation_dataset = FaceDataset(33, 40, root_dir, training_width, training_height)
    validation = create_dataloader(validation_dataset, 1)

    #test_dataloader(dataloader, training_width, training_height)

    net = Net()
    loss = nn.MSELoss()
    opt = Adam(net.parameters(), lr=0.001)
        
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        epoch_loss = torch.zeros((1, 1))
        for i, (images, labels) in enumerate(training):
            prediction = net(images)
            output = loss(prediction, labels[:,-6])
            epoch_loss += output
            output.backward()
            opt.step()
            opt.zero_grad()
        epoch_loss = epoch_loss / len(face_dataset)
        training_losses.append([epoch, epoch_loss.item()*100])

        epoch_loss = torch.zeros((1, 1))
        for i, (images, labels) in enumerate(validation):
            prediction = net(images)
            output = loss(prediction, labels[:,-6])
            epoch_loss += output
            opt.zero_grad()
        epoch_loss = epoch_loss / len(face_dataset)
        validation_losses.append([epoch, epoch_loss.item()*100])


    training_losses = np.array(training_losses)
    validation_losses = np.array(validation_losses)

    plt.plot(training_losses[:,0], training_losses[:,1])
    plt.plot(validation_losses[:,0], validation_losses[:,1])
    plt.plot()
    plt.savefig('results/pt_1/epoch_loss_decrease.png')
    plt.show()



    # with torch.no_grad():
    #     for i, (image, label) in enumerate(dataloader):
    #         prediction = net(image)
    #         output = loss(prediction, label[:,-6])
    #         print("LOSS FOR IMAGE IS: " + str(output))
    #         prediction = prediction.detach().numpy()

    #         plt.imshow(image[0][0], cmap='gray')
    #         plt.scatter(prediction[:,0]*training_width, prediction[:,1]*training_height, s=10, marker='o', c='r')
    #         plt.plot(label[:,-6,0]*training_width, label[:,-6,1]*training_height, marker='o', color='green')
    #         plt.savefig('results/pt_1/prediction_'+str(i)+'_'+str(epochs))
    #         plt.show()



if __name__ == '__main__':
    # Call either of these functions. 
    # pt_1()
    pt_2()


