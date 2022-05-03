from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, num_channels=3):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, model_inputs):
        inp = model_inputs['input']
        x = self.relu(self.conv1(inp))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return {'pred_hr': x}

class ModelPixelShuffle(nn.Module):
    def __init__(self, num_channels=3, scale = 2):
        super(ModelPixelShuffle, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(35, num_channels * self.scale * self.scale, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, model_inputs):
        inp = model_inputs['input']
        x = self.relu(self.conv1(inp))
        x = self.relu(self.conv2(x))
        x = torch.cat([inp, x], axis = 1)
        x = self.conv3(x)
        x = torch.nn.functional.pixel_shuffle(x, self.scale) 
        return {'pred_hr': x}
