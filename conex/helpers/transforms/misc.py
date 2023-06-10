import torch
import torch.nn.functional as F


class SqueezeTransform(torch.nn.Module):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, image):
        return image.squeeze(dim=self.dim)


class SwapTransform(torch.nn.Module):
    def __init__(self, axis1, axis2):
        self.axes = (axis1, axis2)

    def __call__(self, image):
        return image.swapaxes(*self.axes)


class DeviceTransform(torch.nn.Module):
    def __init__(self, device):
        self.device = device

    def __call__(self, image):
        return image.to(self.device)


class Conv2dFilter(torch.nn.Module):
    def __init__(self, filters, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.filters = filters

    def __call__(self, image):
        return F.conv2d(image, self.filters, stride=self.stride, padding=self.padding)
