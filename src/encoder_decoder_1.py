import torch.nn as nn
import torch


class encoder1(nn.Module):
    def __init__(self, vgg1):
        super(encoder1, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg1.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg1.get(0).bias.float())
        self.rp1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg1.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg1.get(2).bias.float())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.rp1(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class decoder1(nn.Module):
    def __init__(self, d1):
        super(decoder1, self).__init__()
        self.rp2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(d1.get(1).weight.float())
        self.conv3.bias = torch.nn.Parameter(d1.get(1).bias.float())

    def forward(self, x):
        out = self.rp2(x)
        out = self.conv3(out)
        return out



