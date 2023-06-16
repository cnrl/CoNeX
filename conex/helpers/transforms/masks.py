import math
import torch
import torchvision.transforms.functional as TF

from itertools import product


class GridEraseMask:
    """
    A Transformer that Grids input and removes only one cell.
    Adding a new dimension in position for total number of cells.
    Inputs should have channel at first index.

    Args:
        m (int): The grid row size.
        n (int): The grid column size.
        random (bool): If true, shuffles the order of the masks.
    """

    def __init__(self, m, n, random=False):
        self.m = m
        self.n = n
        self.random = random

    def __call__(self, img):
        _, h, w = img.shape
        w_grid = math.ceil(w / self.n)
        h_grid = math.ceil(h / self.m)

        result = []
        location = torch.ones(
            self.m * self.n, self.m, self.n, dtype=torch.bool, device=img.device
        )
        for index, ij in enumerate(product(range(self.m), range(self.n))):
            i, j = ij
            result.append(TF.erase(img, i * h_grid, j * w_grid, h_grid, w_grid, v=0))
            location[index, ij[0], ij[1]] = False

        result = torch.stack(result)
        if self.random:
            indices = torch.randperm(result.size(0), device=result.device)
            location = location[indices]
            result = result[indices]
        return result, location


class GridKeepMask:
    """
    A Transformer that Grids input and removes alls but only one cell.
    Adding a new dimension in position for total number of cells.
    Inputs should have channel at first index.

    Args:
        m (int): The grid row size.
        n (int): The grid column size.
        random (bool): If true, shuffles the order of the masks.
    """

    def __init__(self, m, n, random=False):
        self.m = m
        self.n = n
        self.random = random

    def __call__(self, img):
        _, h, w = img.shape
        w_grid = math.ceil(w / self.n)
        h_grid = math.ceil(h / self.m)

        result = []
        location = torch.zeros(
            self.m * self.n, self.m, self.n, dtype=torch.bool, device=img.device
        )
        for index, ij in enumerate(product(range(self.m), range(self.n))):
            i, j = ij
            bg = torch.zeros_like(img)
            bg[:, i * h_grid : (i + 1) * h_grid, j * w_grid : (j + 1) * w_grid] = img[
                :, i * h_grid : (i + 1) * h_grid, j * w_grid : (j + 1) * w_grid
            ]
            result.append(bg)
            location[index, ij[0], ij[1]] = True

        result = torch.stack(result)
        if self.random:
            indices = torch.randperm(result.size(0), device=result.device)
            location = location[indices]
            result = result[indices]
        return result, location


class GridCropMask:
    """
    A Transformer that Grids input and crops to one cell.
    Adding a new dimension in position for total number of cells.
    Inputs should have channel at first index.

    Args:
        m (int): The grid row size.
        n (int): The grid column size.
        random (bool): If true, shuffles the order of the masks.
    """

    def __init__(self, m, n, random=False):
        self.m = m
        self.n = n
        self.random = random

    def __call__(self, img):
        _, h, w = img.shape
        w_grid = math.ceil(w / self.n)
        h_grid = math.ceil(h / self.m)

        result = []
        location = torch.zeros(
            self.m * self.n, self.m, self.n, dtype=torch.bool, device=img.device
        )
        for index, ij in enumerate(product(range(self.m), range(self.n))):
            i, j = ij
            result.append(TF.crop(img, i * h_grid, j * w_grid, h_grid, w_grid))
            location[index, ij[0], ij[1]] = True

        result = torch.stack(result)
        if self.random:
            indices = torch.randperm(result.size(0), device=result.device)
            location = location[indices]
            result = result[indices]
        return result, location
