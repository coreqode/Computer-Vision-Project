import numpy as np
from PIL import Image
import cv2
import h5py
import torch
from torch.utils.data import Dataset
import glob
from natsort import natsorted
import random

class Image91Dataset(Dataset):
    def __init__(self, filepath, scale , patch_size , split = 'train', split_ratio = 0.85):
        self.path = filepath
        self.all_files = natsorted(glob.glob(f'{self.path}/*.png'))
        self.scale = scale
        self.patch_size = patch_size

        if split == 'train':
            self.all_files = self.all_files[: int(split_ratio * len(self.all_files))]
            random.shuffle(self.all_files)
        else:
            self.all_files = self.all_files[int(split_ratio * len(self.all_files)):]

    def __getitem__(self, idx):
        path = self.all_files[idx]
        lr, hr = self.load_and_preprocess(path)
        
        hr = self.normalize(hr)
        lr = self.normalize(lr)

        hr = torch.from_numpy(hr).float().permute((2, 0, 1))
        lr = torch.from_numpy(lr).float().permute((2, 0, 1))

     

        model_inputs = {'input': lr}
        data = {'lr': lr, 'hr': hr}
        return model_inputs, data

    def normalize(self, img):
        img = img / 127.5 - 1
        return img

    def load_and_preprocess(self, path):
        img = Image.open(path).convert('RGB')
        hr_width = (img.width // self.scale) * self.scale
        hr_height = (img.height // self.scale) * self.scale
        hr = img.resize((hr_width, hr_height), resample=Image.BICUBIC)
        lr = hr.resize((hr_width // self.scale, hr_height // self.scale), resample=Image.BICUBIC)
        lr = lr.resize((lr.width * self.scale, lr.height * self.scale), resample=Image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        #hr = convert_rgb_to_y(hr)
        #lr = convert_rgb_to_y(lr)

        ## Randomize the patch size
        i = random.choice(np.arange(lr.shape[0] - self.patch_size))
        j = random.choice(np.arange(lr.shape[1] - self.patch_size))

        lr_patch = lr[i:i + self.patch_size, j:j + self.patch_size]
        hr_patch = hr[i:i + self.patch_size, j:j + self.patch_size]
        return lr_patch, hr_patch

    def convert_rgb_to_y(self, img):
        if type(img) == np.ndarray:
            return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        elif type(img) == torch.Tensor:
            if len(img.shape) == 4:
                img = img.squeeze(0)
            return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        else:
            raise Exception('Unknown Type', type(img))



    def __len__(self):
        return len(self.all_files)

if __name__ == '__main__':
    path = './data/T91'
    dataset = Image91Dataset(path)
    mi, data = dataset[0]
