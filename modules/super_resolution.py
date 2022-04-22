import sys
import os
import numpy as np
from torchsummary import summary
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from base.base_module import BaseModule
from nets.model import Model
from datasets.dataset import Image91Dataset
from utils.loss import *
from utils.metrics import PSNR, SSIM

class SRCNN(BaseModule):
    def __init__(self):
        super().__init__()
        self.epoch = 100
        self.data_dir = "./data/"
        self.num_workers = 8
        self.train_batch_size = 8
        self.val_batch_size = 8
        self.train_shuffle = True
        self.val_shuffle = False
        self.pin_memory = True
        self.split_ratio = 0.85

    def define_dataset(self):
        path = os.path.join(self.data_dir, '91_images_data.h5')

        self.train_dataset = Image91Dataset(path, split = 'train', split_ratio = self.split_ratio)
        self.val_dataset = Image91Dataset(path, split = 'val', split_ratio = self.split_ratio)

    def define_model(self):
        self.model = Model(num_channels = 1)

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
        
def main():
    m = SRCNN()
    m.init(wandb_log=False, project='SuperResolution', entity='noldsoul')
    m.define_model()
    #m.inspect_dataset()
    m.train()


if __name__ == "__main__":
    main()
