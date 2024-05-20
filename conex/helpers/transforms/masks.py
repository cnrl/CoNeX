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
        gap (tuple(int)): left, bottom, up, right gaps for the cell.
    """

    def __init__(self, m, n, random=False, gap=(0, 0, 0, 0)):
        self.m = m
        self.n = n
        self.random = random
        self.gap = gap

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
            h_cor = i * h_grid + self.gap[0]
            dh = h_cor if h_cor < 0 else 0
            w_cor = j * w_grid + self.gap[2]
            dw = w_cor if w_cor < 0 else 0
            result.append(
                TF.erase(
                    img,
                    max(h_cor, 0),
                    max(w_cor, 0),
                    h_grid - self.gap[1] - self.gap[2] + dh,
                    w_grid - self.gap[3] - self.gap[0] + dw,
                    v=0,
                )
            )
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
        gap (tuple(int)): left, bottom, up, right gaps for the cell.
    """

    def __init__(self, m, n, random=False, gap=(0, 0, 0, 0)):
        self.m = m
        self.n = n
        self.random = random
        self.gap = gap

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
            bg[
                :,
                max(i * h_grid + self.gap[0], 0) : min(
                    (i + 1) * h_grid - self.gap[3], img.size(1)
                ),
                max(j * w_grid + self.gap[2], 0) : min(
                    (j + 1) * w_grid - self.gap[1], img.size(2)
                ),
            ] = img[
                :,
                max(i * h_grid + self.gap[0], 0) : min(
                    (i + 1) * h_grid - self.gap[3], img.size(1)
                ),
                max(j * w_grid + self.gap[2], 0) : min(
                    (j + 1) * w_grid - self.gap[1], img.size(2)
                ),
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
        gap (tuple(int)): left, bottom, up, right gaps for the cell.
    """

    def __init__(self, m, n, random=False, gap=(0, 0, 0, 0)):
        self.m = m
        self.n = n
        self.random = random
        self.gap = gap

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
            result.append(
                TF.crop(
                    img,
                    i * h_grid + self.gap[0],
                    j * w_grid + self.gap[2],
                    h_grid - self.gap[1] - self.gap[2],
                    w_grid - self.gap[3] - self.gap[0],
                )
            )
            location[index, ij[0], ij[1]] = True

        result = torch.stack(result)
        if self.random:
            indices = torch.randperm(result.size(0), device=result.device)
            location = location[indices]
            result = result[indices]
        return result, location
