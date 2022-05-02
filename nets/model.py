from torch import nn

class Model(nn.Module):
    def __init__(self,  n1=64, n2=32, f1=9, f2=5, f3=5, n3=0, f4=5, num_channels=3):
        super(Model, self).__init__()
        self.n3 = n3
        self.conv1 = nn.Conv2d(num_channels, n1, kernel_size=f1, padding=f1 // 2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2 // 2)
        if n3>0:
            self.conv3 = nn.Conv2d(n2, n3, kernel_size=f3, padding=f3 // 2)
            self.conv4 = nn.Conv2d(n3, num_channels, kernel_size=f4, padding=f4 // 2)
        else:
            self.conv3 = nn.Conv2d(n2, num_channels, kernel_size=f3, padding=f3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, model_inputs):
        inp = model_inputs['input']
        x = self.relu(self.conv1(inp))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        if self.n3>0:
            x = self.relu(x)
            x = self.conv4(x)
        return {'pred_hr': x}
