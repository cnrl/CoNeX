import torch


class transform_squeeze(torch.nn.Module):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, image):
        return image.squeeze(dim=self.dim)


class poisson(torch.nn.Module):
    def __init__(self, timesteps, ratio):
        self.timesteps = timesteps
        self.ratio = ratio

    def __call__(self, img):
        random_probability = torch.rand(size=(self.timesteps, *img.shape))
        intensity = img.unsqueeze(dim=0).expand(self.timesteps, *img.shape)
        spike_probaility = intensity * self.ratio
        return spike_probaility >= random_probability


class latency_to_intensity(torch.nn.Module):
    def __init__(self, timesteps, threshold=None, sparsity=None, min_val=0.0, max_val=1.0, lower_trim=True, higher_trim=True):
        self.timesteps = timesteps
        self.threshold = threshold if threshold is not None else min_val
        self.sparsity = sparsity
        self.interval = (min_val, max_val)
        self.higher_trim = higher_trim
        self.lower_trim = lower_trim

    def __call__(self, img):
        self.threshold = img.quantile(
            1 - self.sparsity) if self.sparsity is not None else self.threshold
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

        index = img * max_factor * (self.timesteps-1)
        index = index.ceil().long()
        index += 1
        index[below_index] = 0
        index = index.clamp(0)
        index = index.unsqueeze(0)

        spikes = torch.zeros(self.timesteps+1, *img.shape,
                             dtype=torch.bool, device=img.device)
        spikes.scatter_(0, index, True)
        spikes = spikes[1:]

        return spikes.flip(0)
