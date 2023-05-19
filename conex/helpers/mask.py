import random
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms

class Mask:
    """
        The input must be an image represented by
        and torch tensor. """

    def __init__(self, m, n, mode='erase', keep_base_image = True):
        """
            Arguments:
                m(int) : the rows size
                n(int) : the colemn size
                mode('erase' or 'crop') : the function to do
                keep_base_image(True or Flase) : keep the base image at beginning """
        self.m = m
        self.n = n
        self.mode = mode
        self.keep_base_image = keep_base_image

    def __call__(self, img):
        _, w, h = img.shape
        cell_size_w = w // self.n
        cell_size_h = h // self.m
        # create a collection of images
        collection = img
        
        for i in range(self.n):
            for j in range(self.m):

                box = (i * cell_size_w, j * cell_size_h, (i + 1) * cell_size_w, (j + 1) * cell_size_h)

                if self.mode == 'erase':
                    # Create a black PIL image of the same size as the input image
                    mask = TF.erase(img, i=j*cell_size_h, j=i*cell_size_w, w=cell_size_w, h=cell_size_h, v=0)
                    # add to collection
                    collection = torch.cat((collection,mask), 0)

                elif self.mode == 'crop':
                    # Create a black PIL image of the same size as the input image
                    mask = TF.to_pil_image(torch.zeros_like(img))
                    # Paste the selected cell onto the black image
                    mask.paste(TF.to_pil_image(img).crop(box), box)
                    # add to collection
                    collection = torch.cat((collection,TF.to_tensor(mask)), 0)

        if self.keep_base_image == False:
            collection = collection[3:]
        return collection


# E.G.
# mnist_testset = datasets.CIFAR10(root='./data', train=False, download=True, 
#                                transform=transforms.Compose([transforms.ToTensor(),
#                                                              Mask(m=2, n=2, mode='erase', keep_base_image = True),                                                       
#                                                             ]))
# print(f" shape must be (15, 32, 32), becuase we add m*n*3 more layers(photos)")
# print(f"shape is : {mnist_testset[0][0].shape}")
# print(f"prinring the original image([0:3] represent the original image)")
# plt.imshow(mnist_testset[0][0][0:3].permute(1, 2, 0))