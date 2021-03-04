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


class encoder2(nn.Module):
    def __init__(self, vgg):
        super(encoder2, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
        self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.rp1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        self.rp3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.rp4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.rp1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.rp3(out)
        out = self.conv3(out)
        pool = self.relu3(out)
        out, pool_idx = self.mp(pool)
        out = self.rp4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        return out
