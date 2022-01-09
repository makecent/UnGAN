import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch


class CaptchaDataset(Dataset):
    def __init__(self, img_dir, transform=None, **kwargs):
        super(CaptchaDataset, self).__init__(**kwargs)

        self.img_dir = img_dir
        self.transform = transform
        self.num = 8229

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{idx:05d}.jpg')
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 1

    @staticmethod
    def encode_label(target):
        encoded = torch.zeros(4, 10)
        for i, v in enumerate(f'{target:04d}'):
            v = int(v)
            encoded[i][v] = 1
        return encoded.flatten()

    @staticmethod
    def decode_label(target):
        decoded = target.reshape(4, 10)
        for i in decoded:
            v = i.argmax()
            decoded[v][i] = 1
        return decoded.flatten()