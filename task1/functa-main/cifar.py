import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import utils

class CIFAR(Dataset):
    """
    Toy Demo
    """

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
    ) -> None:
        super().__init__()
        self.data = root
        self.transforms = transform
        self.train = train
        samples = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

        x_load = np.stack([sample[0] for sample in samples])
        y_load = np.stack([sample[1] for sample in samples])


        # self.x = x_load / 255.
        self.x = x_load
        self.y = y_load
        # pad_amount = ((0,0), (0, 0), (2, 2), (2, 2))
        # self.x = np.pad(self.x, pad_amount, 'constant')

        # let's get some shapes to understand what we loaded.
        print('shape of X: {}, y: {}'.format(self.x.shape, self.y.shape))

    def __getitem__(self, idx):
        """
        get item
        """
        img = self.x[idx]
        label = self.y[idx]

        coord = utils.get_coordinate_grid(img.shape[1]).astype("float32")
        return img, coord, label, idx

    def __len__(self):
        return len(self.x)