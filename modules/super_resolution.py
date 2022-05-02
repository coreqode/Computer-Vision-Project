import sys
import time
import os
import numpy as np
from torchsummary import summary
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from base.base_module import BaseModule
from nets.model import Model
from datasets.dataset import Image91Dataset, set5Dataset
from utils.loss import *
from utils.metrics import PSNR, SSIM
import argparse

class SRCNN(BaseModule):
    def __init__(self):
        super().__init__()
        self.epoch = 50
        self.data_dir = "./data/"
        self.num_workers = 8
        self.train_batch_size = 8
        self.val_batch_size = 8
        self.train_shuffle = True
        self.val_shuffle = False
        self.pin_memory = True
        self.split_ratio = 0.85

    def define_dataset(self):
        path = os.path.join(self.data_dir, '91-image_x4.h5')

        self.train_dataset = Image91Dataset(path, split = 'train', split_ratio = self.split_ratio)
        self.val_dataset = Image91Dataset(path, split = 'val', split_ratio = self.split_ratio)

        path_test = os.path.join(self.data_dir, 'Set5_x4.h5')
        self.test_dataset = set5Dataset(path_test)

        self.train_dataset = Image91Dataset(path, split = 'train', split_ratio = self.split_ratio)
        self.val_dataset = Image91Dataset(path, split = 'val', split_ratio = self.split_ratio)

    def define_model(self, n1=64, n2=32, f1=9, f2=5, f3=5, n3=0, f4=5, num_channels=1):
        self.model = Model(n1, n2, f1, f2, f3, n3, f4, num_channels = 1 )

    def define_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def loss_func(self, data, predictions):
        ### implement loss
        ml = mse_loss(predictions['pred_hr'], data['hr'])
        loss = {"mse_loss": ml}
        return loss

    def inspect_dataset(self):
        for idx, (model_inputs, data) in enumerate(self.train_loader):
            print(model_inputs)
            break

    def calc_psnr(self, img1, img2):
        psnr = PSNR()
        return psnr(img1, img2)

    def calc_ssim(self, img1, img2):
        ssim = SSIM()
        return ssim(img1, img2)

    def inspect_dataset(self):
        for idx, (model_inputs, data) in enumerate(self.train_loader):
            print(model_inputs)
            break

    def inspect_test_dataset(self):
        for idx, (model_inputs, data) in enumerate(self.test_loader):

            for id, inimg in enumerate(data['lr']):
                plt.imsave('./643232/inference_x4/lr_in'+str(idx)+str(id)+'.png', inimg[0].numpy(), cmap='gray')

            for id, inimg in enumerate(data['hr']):
                plt.imsave('./643232/inference_x4/hr_in'+str(idx)+str(id)+'.png', inimg[0].numpy(), cmap='gray')

        
def main():
    m = SRCNN()
    m.init(wandb_log=True, project='SuperResolution_1117', entity='neha-s')
    m.define_model(n1=64, n2=32, n3 = 32, f1 = 9, f2=3, f3=3, f4=5)
    #m.inspect_dataset()
    m.train()
    m.inspect_test_dataset()
    m.save_output_imgs()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('n1', type=int, help='n1', default=64)
    # parser.add_argument('n2', type=int, default=32,help='n2')
    # parser.add_argument('f1', type=int, help='f1', default=9)
    # parser.add_argument('f2', type=int, help='f2', default=5)
    # parser.add_argument('f3', type=int, help='f3', default=5)
    # parser.add_argument('num_layers', type=int, help='num_layers', default=2)
    
    # args = parser.parse_args()
    # if args.num_layers>2:
    #     n1, n2, n3 = input('enter n1, n2, n3 : ')
    main()
