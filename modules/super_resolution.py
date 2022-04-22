import sys
import numpy as np
from torchsummary import summary
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from base.base_module import BaseModule

class SRCNN(BaseModule):
    def __init__(self):
        super().__init__()
        self.epoch = 100
        self.data_dir = "./data/"
        self.num_workers = 0
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.prefetch_factor = 2
        self.train_shuffle = True
        self.val_shuffle = False
        self.pin_memory = False

    def define_dataset(self):
        self.train_dataset =  FlagSimpleDataset(device=self.device, 
                                    path='./data/flag_simple', history = True , 
                                    split='train', node_info=self.node_info, 
                                    augmentation = True)
        
        self.val_dataset =  FlagSimpleDataset(device=self.device, 
                                    path='./data/flag_simple', history = True , 
                                    split='train', node_info=self.node_info, 
                                    augmentation = True)


    def define_model(self):
        self.model = Model(self.device, size =3)

    def loss_func(self, data, predictions):
        ### implement loss
        loss = {"mse_loss": mse_loss}
        return loss

    def inspect_dataset(self):
        for idx, (model_inputs, data) in enumerate(self.train_loader):
            model_inputs, data = self.send_to_cuda(model_inputs[0], data[0])
            out = self.model(model_inputs, is_training = True)
            print(out)
            break
        
def main():
    m = SRCNN()
    m.init(wandb_log=False, project='SuperResolution', entity='noldsoul')
    m.define_model()
    # h.inspect_dataset()
    m.train()


if __name__ == "__main__":
    main()
