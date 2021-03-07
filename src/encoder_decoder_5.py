import torch.nn as nn
import torch


class encoder5(nn.Module):
    def __init__(self, vgg):
        super(encoder5, self).__init__()
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
        self.rp5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.rp6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        self.rp7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv7.weight = torch.nn.Parameter(vgg.get(19).weight.float())
        self.conv7.bias = torch.nn.Parameter(vgg.get(19).bias.float())
        self.relu7 = nn.ReLU(inplace=True)
        self.rp8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv8.weight = torch.nn.Parameter(vgg.get(22).weight.float())
        self.conv8.bias = torch.nn.Parameter(vgg.get(22).bias.float())
        self.relu8 = nn.ReLU(inplace=True)
        self.rp9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv9.weight = torch.nn.Parameter(vgg.get(25).weight.float())
        self.conv9.bias = torch.nn.Parameter(vgg.get(25).bias.float())
        self.relu9 = nn.ReLU(inplace=True)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.rp10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv10.weight = torch.nn.Parameter(vgg.get(29).weight.float())
        self.conv10.bias = torch.nn.Parameter(vgg.get(29).bias.float())
        self.relu10 = nn.ReLU(inplace=True)
        self.rp11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv11.weight = torch.nn.Parameter(vgg.get(32).weight.float())
        self.conv11.bias = torch.nn.Parameter(vgg.get(32).bias.float())
        self.relu11 = nn.ReLU(inplace=True)
        self.rp12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv12.weight = torch.nn.Parameter(vgg.get(35).weight.float())
        self.conv12.bias = torch.nn.Parameter(vgg.get(35).bias.float())
        self.relu12 = nn.ReLU(inplace=True)
        self.rp13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv13.weight = torch.nn.Parameter(vgg.get(38).weight.float())
        self.conv13.bias = torch.nn.Parameter(vgg.get(38).bias.float())
        self.relu13 = nn.ReLU(inplace=True)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.rp14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv14.weight = torch.nn.Parameter(vgg.get(42).weight.float())
        self.conv14.bias = torch.nn.Parameter(vgg.get(42).bias.float())
        self.relu14 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.rp1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.rp3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out, pool_idx = self.mp(out)
        out = self.rp4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.rp5(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out, pool_idx2 = self.mp2(out)
        out = self.rp6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.rp7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.rp8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.rp9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out, pool_idx3 = self.mp3(out)
        out = self.rp10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.rp11(out)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.rp12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.rp13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out, pool_idx4 = self.mp4(out)
        out = self.rp14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        return out


class decoder5(nn.Module):
    def __init__(self, d):
        super(decoder5, self).__init__()
        self.rp15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
        self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.rp16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
        self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        self.rp17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
        self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        self.rp18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
        self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        self.rp19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
        self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.rp20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
        self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        self.rp21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
        self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)
        self.rp22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
        self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)
        self.rp23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
        self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.rp24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
        self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)
        self.rp25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
        self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)
        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.rp26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
        self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)
        self.rp27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)
        self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
        self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x):
        out = self.rp15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.rp16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.rp17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.rp18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.rp19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.rp20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.rp21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.rp22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.rp23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.rp24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.rp25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.rp26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.rp27(out)
        out = self.conv27(out)
        return out
