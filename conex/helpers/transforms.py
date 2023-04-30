import torch


class SqueezeTransform(torch.nn.Module):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, image):
        return image.squeeze(dim=self.dim)
