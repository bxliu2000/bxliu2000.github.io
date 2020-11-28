from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.optim import Adam
import cv2

from dataset import FaceDataset, create_dataloader, test_dataloader
from transforms import CustomTransforms

root_dir = './imm_face_db/'

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 12->20->28 output channels, 5x5 square convolution
        # kernel
        self.conv = [
                    nn.Conv2d(1, 20, 5), nn.Conv2d(20, 20, 5), 
                    nn.Conv2d(20, 40, 5), nn.Conv2d(40, 60, 3)
                    ]

        self.fc1 = nn.Linear(7560, 640)
        self.fc2 = nn.Linear(640, 116)

        self.max_layers = [0, 1, 2]

    def forward(self, x):
        for i in range(len(self.conv)):
            if i in self.max_layers:
                x = F.max_pool2d(F.relu(self.conv[i](x)), 2)
            else:
                x = F.relu(self.conv[i](x))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def pt_2():
    epochs = 25
    H = 120
    W = 160
    
    face_dataset = FaceDataset(1, 32, root_dir, W, H, CustomTransforms())
    training = create_dataloader(face_dataset, 5)

    validation_dataset = FaceDataset(33, 40, root_dir, W, H, CustomTransforms())
    validation = create_dataloader(validation_dataset, 1)

    #test_dataloader(training, W, H)

    net = Net()
    loss = nn.MSELoss()
    opt = Adam(net.parameters(), lr=0.001)
        
    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        epoch_loss = torch.zeros((1, 1))
        for i, (images, labels) in enumerate(training):
            prediction = net(images)
            output = loss(prediction, labels.type(torch.float32).view(-1, 116))
            epoch_loss += output
            output.backward()
            opt.step()
            opt.zero_grad()
        epoch_loss = epoch_loss / len(face_dataset)
        print("EPOCH " + str(i)+ " LOSS: " + str(epoch_loss))
        training_losses.append([epoch, epoch_loss.item()*100])

        epoch_loss = torch.zeros((1, 1), requires_grad=False)
        for i, (images, labels) in enumerate(validation):
            prediction = net(images)
            output = loss(prediction, labels.type(torch.float32).view(-1, 116))
            epoch_loss += output
            opt.zero_grad()
        epoch_loss = epoch_loss / len(face_dataset)
        validation_losses.append([epoch, epoch_loss.item()*100])


    training_losses = np.array(training_losses)
    validation_losses = np.array(validation_losses)

    plt.plot(training_losses[:,0], training_losses[:,1])
    plt.plot(validation_losses[:,0], validation_losses[:,1])
    plt.plot()
    plt.savefig('results/pt_2/epoch_loss_decrease.png')
    plt.show()

    """
    Handy visualization code copied and pasted from:
    https://colab.research.google.com/github/Niranjankumar-c/DeepLearning-PadhAI/blob/master/DeepLearning_Materials/6_VisualizationCNN_Pytorch/CNNVisualisation.ipynb#scrollTo=cWmfCalUvzbS
    as linked on the piazza. 
    """
    def plot_filters_single_channel(i, t):
        
        #kernels depth * number of kernels
        nplots = t.shape[0]*t.shape[1]
        ncols = 12
        
        nrows = 1 + nplots//ncols
        #convert tensor to numpy image
        npimg = np.array(t.numpy(), np.float32)
        
        count = 0
        fig = plt.figure(figsize=(ncols, nrows))
        
        #looping through all the kernels in each channel
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                count += 1
                ax1 = fig.add_subplot(nrows, ncols, count)
                npimg = np.array(t[i, j].numpy(), np.float32)
                npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                ax1.imshow(npimg)
                ax1.set_title(str(i) + ',' + str(j))
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
       
        plt.tight_layout()
        plt.savefig(str(i) + 'weight_visualization.png')
        plt.show()

    for i in range(len(net.conv)):
        if i == 0:
            plot_filters_single_channel(i, net.conv[i].weight.data)


    validation_dataset = FaceDataset(33, 40, root_dir, W, H, CustomTransforms())
    dataloader = create_dataloader(validation_dataset, 1)

    with torch.no_grad():
        for i, (image, label) in enumerate(dataloader):
            prediction = net(image)
            output = loss(prediction, label.type(torch.float32).view(-1, 116))
            print("LOSS FOR IMAGE IS: " + str(output))
            
            prediction = prediction.view(-1, 58, 2)

            plt.imshow(image[0][0], cmap='gray')
            plt.scatter(prediction[0,:,0]*W, prediction[0,:,1]*H, s=10, marker='o', c='r')
            plt.scatter(label[0,:,0]*W, label[0,:,1]*H, marker='o', color='green')
            plt.savefig('results/prediction_'+str(i)+'_'+str(epochs))

            plt.show()



