import torch
import torchvision.transforms.functional as TF


class Mask:
    """
    The input must be an image represented by
    and torch tensor."""

    def __init__(self, m, n, mode="erase", keep_base_image=True):
        """
        Arguments:
            m(int) : the rows size
            n(int) : the colemn size
            mode('erase' or 'crop') : the function to do
            keep_base_image(True or Flase) : keep the base image at beginning"""
        self.m = m
        self.n = n
        self.mode = mode
        self.keep_base_image = keep_base_image

    def __call__(self, img):
        _, w, h = img.shape
        cell_size_w = w // self.n
        cell_size_h = h // self.m
        collection = img

        for i in range(self.n):
            for j in range(self.m):
                box = (
                    i * cell_size_w,
                    j * cell_size_h,
                    (i + 1) * cell_size_w,
                    (j + 1) * cell_size_h,
                )

                if self.mode == "erase":
                    mask = TF.erase(
                        img,
                        i=j * cell_size_h,
                        j=i * cell_size_w,
                        w=cell_size_w,
                        h=cell_size_h,
                        v=0,
                    )
                    collection = torch.cat((collection, mask), 0)

                elif self.mode == "crop":
                    mask = TF.to_pil_image(torch.zeros_like(img))
                    mask.paste(TF.to_pil_image(img).crop(box), box)
                    collection = torch.cat((collection, TF.to_tensor(mask)), 0)

        if not self.keep_base_image:
            collection = collection[3:]
        return collection
