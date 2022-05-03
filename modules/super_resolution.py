import sys
import os
import numpy as np
from torchsummary import summary
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from base.base_module import BaseModule
from nets.model import Model, ModelPixelShuffle
from datasets.dataset import Image91Dataset
from utils.loss import *
from utils.metrics import PSNR, SSIM

class SRCNN(BaseModule):
    def __init__(self):
        super().__init__()
        self.epochs = 800
        self.data_dir = "./data/"
        self.num_workers = 8
        self.train_batch_size = 8
        self.val_batch_size = 8
        self.train_shuffle = True
        self.val_shuffle = False
        self.pin_memory = True
        self.split_ratio = 0.85
        self.scale = 2
        self.bicubic = True
        self.patch_size = 32

    def define_dataset(self):
        path = os.path.join(self.data_dir, 'T91')

        self.train_dataset = Image91Dataset(path, scale = self.scale, bicubic = self.bicubic, patch_size = self.patch_size, split = 'train', split_ratio = self.split_ratio)
        self.val_dataset = Image91Dataset(path, scale = self.scale, bicubic = self.bicubic, patch_size = self.patch_size, split = 'val', split_ratio = self.split_ratio)

    def define_model(self):
        #self.model = ModelPixelShuffle(scale  = self.scale)
        self.model = Model()

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

    def ssim(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def inference(self):
        def calc_psnr(img1, img2):
            return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

        import glob
        from PIL import Image
        import os

        test_images = glob.glob('./data/Set5/GTmod12/*.png')
        out_path = f'./inference/T91_RGB_scale_{self.scale}'
        os.makedirs(out_path, exist_ok = True)
        self.model = self.model.to(self.device)
        chk = torch.load(f'./weights_{self.scale}_T91/model_620.pt')
        self.model.load_state_dict(chk)
        
        with open(os.path.join(out_path, 'metrics.txt'), 'w') as fi:
            for path in test_images:
                image = Image.open(path).convert('RGB')
                image_width = (image.width // self.scale) * self.scale
                image_height = (image.height // self.scale) * self.scale
                hr = image.resize((image_width, image_height), resample=Image.BICUBIC)
                image = hr.resize((image.width // self.scale, image.height // self.scale), resample=Image.BICUBIC)
                if self.bicubic:
                    image = image.resize((image.width * self.scale, image.height * self.scale), resample=Image.BICUBIC)
                name = path.split('/')[-1].split('.')[0]
                bicubic_path = os.path.join(out_path, f'{name}_bicubic.png')
                hr_path = os.path.join(out_path, f'{name}_hr.png')
                hr.save(hr_path)
                image.save(bicubic_path)
                image = np.array(image).astype(np.float32)
                inputs = torch.from_numpy(image).to(self.device)
                inputs = inputs / 127.5 - 1 
                inputs = inputs.permute((2,0,1)).unsqueeze(0)
                model_inputs = {'input':inputs}

                with torch.no_grad():
                    preds = self.model(model_inputs)

                pred = preds['pred_hr'][0].permute((1,2,0))
                pred = torch.clamp(pred, -1,1)
                pred = ((pred + 1 ) * 127.5).detach().cpu()
                psnr = calc_psnr((torch.from_numpy(np.array(hr).astype(np.uint8)).float()) / 255, pred / 255)
                
                fi.write(f"PSNR for {name}: {psnr} \n")
                print(f"PSNR for {name}: ", psnr)
                pred = pred.numpy().astype(np.uint8)

                ssim = self.ssim(pred, np.array(hr).astype(np.uint8))
                fi.write(f"SSIM for {name}: {ssim} \n")
                print(f"SSIM for {name}: ", ssim)
                pred = Image.fromarray(pred)
                pred_path = os.path.join(out_path, f'{name}_output.png')
                pred.save(pred_path)
        
def main():
    m = SRCNN()
    m.init(wandb_log=False, project='SuperResolution', entity='noldsoul')
    m.define_model()
    #m.inspect_dataset()
    #m.train()
    m.inference()


if __name__ == "__main__":
    main()
