import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_ch=10):
        super(InceptionBlock, self).__init__()

        self.branch_1x1 = nn.Conv2d(in_channels, out_ch, kernel_size=1)
        self.branch_3x3 = nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=1)
        self.branch_5x5 = nn.Conv2d(in_channels, out_ch, kernel_size=5, padding=2)


    def forward(self, x):
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        branch_5x5 = self.branch_5x5(x)
        outputs = [branch_1x1, branch_3x3, branch_5x5]
        return torch.cat(outputs, 1)


class Net(nn.Module):
    def __init__(self, in_channels, out_ch):
        super(Net, self).__init__()

        self.inception_1 = nn.Sequential(
            InceptionBlock(in_channels),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )
        self.inception_2 = nn.Sequential(
            InceptionBlock(30),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(30*8*8, 256),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, out_ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.inception_1(x)
        x = self.inception_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x