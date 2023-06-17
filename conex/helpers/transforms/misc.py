import torch
import torch.nn.functional as F


class UnsqueezeTransform(torch.nn.Module):
    """
    Apply pytorch unsqueeze function.

    Args:
        dim (int): The index to insert new dimension
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, input):
        return input.unsqueeze(dim=self.dim)


class SqueezeTransform(torch.nn.Module):
    """
    Apply pytorch squeeze function.

    Args:
        dim (int): The index to remove a singleton dimension. Ff None all singleton dimensions will be removed.
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, input):
        return input.squeeze(dim=self.dim)


class SwapTransform(torch.nn.Module):
    """
    Transpose two axes of input.

    Args:
        axis1 (int): The first dimension to be transposed.
        axis2 (int): The second dimension to be transposed.
    """

    def __init__(self, axis1, axis2):
        self.axes = (axis1, axis2)

    def __call__(self, input):
        return input.swapaxes(*self.axes)


class DeviceTransform(torch.nn.Module):
    """
    Transfer data to new device.

    Args:
        device (str): Device where data will located on.
    """

    def __init__(self, device):
        self.device = device

    def __call__(self, input):
        return input.to(device=self.device)


class Conv2dFilter(torch.nn.Module):
    """
    Convolve a filter on data.

    Args:
        filter (str): Filter to convolve data with.
        stride (int or tuple): Stride of the convolution. the default is 1.
        padding (int or tuple): Padding added to all four sides of the input. the default is 0.
    """

    def __init__(self, filters, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.filters = filters

    def __call__(self, input):
        return F.conv2d(input, self.filters, stride=self.stride, padding=self.padding)


class AbsoluteTransform(torch.nn.Module):
    """
    Apply pytorch absolute function on data.
    """

    def __call__(self, input):
        return torch.abs(input)


class DivideSignPolarity(torch.nn.Module):
    """
    Divide positive and negative values.
    The transformed data will have double size for the first dimension with first have all negative values changed to zero and
    second half with the absolute of the negative values.
    """

    def __call__(self, input):
        p_input = input * (input > 0)
        n_input = -input * (input < 0)
        return torch.cat([p_input, n_input], axis=0)
