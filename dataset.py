from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

"""
Main Dataset class for training and evaluation. The start and end
parameters aare the indices of the photographs in our dataset. E.g. 
a start of 1 and an end of 32 means we want this dataset to include
image 1 through image 32 inclusive in this set. 

For parts 1 and 2, we can specify various height and widths.

For part 2, we can apply custom transforms for data augmentation.
"""
class FaceDataset(Dataset):

    def __init__(self, start, end, root_dir, \
                im_width, im_height, \
                transform=None):
        self.landmarks = self.read_landmarks(start, end, root_dir)
        self.root_dir = root_dir
        self.W = im_width
        self.H = im_height
        self.transform = transform

    def read_landmarks(self, start, end, root_dir):
        landmarks = []

        # Loop through the images we want.
        for i in range(start, end + 1):
            # Loop through all different angles. 
            for j in range(1,7):
                # Construct name of the asf file.
                im_path = root_dir + '{:02d}-{:d}{}'.format(i,j,'m')
                if not os.path.exists(im_path + '.asf'):
                    im_path = root_dir + '{:02d}-{:d}{}'.format(i,j,'f')

                # Actually read the file in. 
                file = open(im_path + '.asf')
                points = file.readlines()[16:74]
                landmark = [im_path + '.jpg']
                for point in points:
                    x,y = point.split('\t')[2:4]
                    landmark.append(float(x))
                    landmark.append(float(y)) 

                # # the nose keypoint
                # nose_keypoint = np.array(landmark).astype('float32')[-6]
                landmarks.append(landmark)
        return np.array(landmarks)


    def __len__(self):
        return len(self.landmarks)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.landmarks[idx, 0]
        image = io.imread(img_name)
        labels = np.array([self.landmarks[idx, 1:]]).astype('float32').reshape(-1, 2)
        image = transforms.ToPILImage()(image)
        if self.transform:
            image, labels = self.transform(image, labels)
        #greyscale, resize, then make it a tensor.
        image = TF.to_grayscale(image)
        image = TF.resize(image, (self.H, self.W))
        image = TF.to_tensor(image)

        #normalize
        for i in range(len(image)):
            image[i,:,:] = image[i,:,:] / torch.max(image[i])

        image[:,:,:] = image[:,:,:] - 0.5

        return (image, labels)

"""
Wrapper function to help us get a dataloader. 
"""
def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

"""
Helper function to display a batch. 
"""
def show_landmarks_batch(sample_batched, W, H):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched[0], sample_batched[1]
    batch_size = len(images_batch)
    print(images_batch.shape)
    im_size = images_batch.size(-1)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)), cmap='gray')

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy()*W + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy()*H + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

"""
Actual function called to test a dataloader.
"""
def test_dataloader(dataloader, W, H):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched[0].size(),
              sample_batched[1].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched, W, H)
            plt.axis('off')
            plt.ioff()
            plt.show()

            break


