import torch


class Poisson(torch.nn.Module):
    """
    Simple Poisson encoding.
    Input values should be between 0 and 1. Spike rate is increased linearly with regard to the values.
    This transformer uses regular random generator provided for `torch.rand`.

    Args:
        time_window (int): The interval of the coding.
        ratio (float): A scale factor for probability of spiking.
    """

    def __init__(self, time_window, ratio):
        self.time_window = time_window
        self.ratio = ratio

    def __call__(self, img):
        random_probability = torch.rand(size=(self.time_window, *img.shape))
        intensity = img.unsqueeze(dim=0).expand(self.time_window, *img.shape)
        spike_probability = intensity * self.ratio
        return spike_probability >= random_probability


class Intensity2Latency(torch.nn.Module):
    """
    Intensity to latency encoding.
    Stronger values spikes sooner.

    Args:
        time_windows (int): The interval of the coding.
        threshold (float): If not None, values lower than threshold will not spike.
        sparsity (float): If not None, defines a threshold for each input based on sparsity.
        min_val (float): Minimum possible value of input. The default is 0.0.
        max_val (float): Maximum possible value of input. The default is 1.0.
        lower_trim (bool): If True, spikes are transformed in order to have the last spike on the end of the interval. The default is True.
        higher_trim (bool): If True, spikes are transformed in order to have the first spike on the first of the interval. The default is True.
    """

    def __init__(
        self,
        time_window,
        threshold=None,
        sparsity=None,
        min_val=0.0,
        max_val=1.0,
        lower_trim=True,
        higher_trim=True,
    ):
        self.time_window = time_window
        self.threshold = threshold if threshold is not None else min_val
        self.sparsity = sparsity
        self.interval = (min_val, max_val)
        self.higher_trim = higher_trim
        self.lower_trim = lower_trim

    def __call__(self, img):
        self.threshold = (
            img.quantile(1 - self.sparsity)
            if self.sparsity is not None
            else self.threshold
        )
        below_index = img < self.threshold

        img -= self.interval[0]
        max_value = self.interval[1] - self.interval[0]
        max_factor = 1 / max_value

        if self.lower_trim and not below_index.all():
            img_min = img[~below_index].min()
            img = img - img_min
            max_factor = 1 / (max_value - img_min)

        if self.higher_trim and img.max() != 0:
            max_factor = 1 / img.max()

        index = img * max_factor * (self.time_window - 1)
        index = index.ceil().long()
        index += 1
        index[below_index] = 0
        index = index.clamp(0)
        index = index.unsqueeze(0)

        spikes = torch.zeros(
            self.time_window + 1, *img.shape, dtype=torch.bool, device=img.device
        )
        spikes.scatter_(0, index, True)
        spikes = spikes[1:]

        return spikes.flip(0)
