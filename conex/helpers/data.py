from torch.utils.data import Dataset


class LocationDataset(Dataset):
    """
    A custom dataset class for data with triple image, location, label nature.

    Args:
        dataset (Dataset): An Instance of a dataset
        pre_transform (Transform): A transformation to apply on images. If given, transformation should return a `(image, location)` tuple.
        post_transform  (Transform): A Transformation that applies on images. Suitable for encodings.
        location_transform (Transform): A Transformation applies on location data. Suitable for encodings.
        target_transform (Transform): A Transformation applies on labels.
    """

    def __init__(
        self,
        dataset,
        pre_transform=None,
        post_transform=None,
        location_transform=None,
        target_transform=None,
    ):
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.location_transform = location_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        location = None
        if self.pre_transform:
            image, location = self.pre_transform(image)
        if self.post_transform:
            image = self.post_transform(image)
        if self.location_transform:
            location = self.location_transform(location)
        if self.target_transform:
            label = self.target_transform(label)
        return image, location, label
