import numpy as np
from PIL import Image
import cv2
import h5py
import torch
from torch.utils.data import Dataset

class Image91Dataset(Dataset):
    def __init__(self, filepath):
        self.path = filepath

        ## TODO: Write the code for generating the dataset or maybe include here only. 
        with h5py.File(self.path, 'r') as f:
            self.hr = np.array(f['hr'])
            self.lr = np.array(f['lr'])

    def __getitem__(self, idx):
        hr = self.hr[idx]
        lr = self.lr[idx]

        hr = self.normalize(hr)
        lr = self.normalize(lr)

        hr = torch.from_numpy(hr).float().unsqueeze(0)
        lr = torch.from_numpy(lr).float().unsqueeze(0)
        model_inputs = {'input': lr}
        data = {'lr': lr, 'hr': hr}
        return model_inputs, data

    def normalize(self, img):
        img = img / 127.5 - 1
        return img

    def __len__(self):
        return self.lr.shape[0]

if __name__ == '__main__':
    path = '../data/91_images_data.h5'
    dataset = Image91Dataset(path)
    print(dataset[0])
