import torch.nn as nn
import torch


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


class decoder2(nn.Module):
    def __init__(self, d):
        super(decoder2, self).__init__()
        self.rp5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv5.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.rp6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv6.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        self.rp7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv7.bias = torch.nn.Parameter(d.get(8).bias.float())

    def forward(self, x):
        out = self.rp5(x)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.unpool(out)
        out = self.rp6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.rp7(out)
        out = self.conv7(out)
        return out
