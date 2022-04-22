from torch import nn

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
