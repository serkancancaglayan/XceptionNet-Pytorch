import torch
import torch.nn as nn   

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.sep_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.sep_conv(x)
    
class BasicConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)
    

class SeperableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, use_relu = True):
        super(SeperableConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
        )    
        self.relu = nn.ReLU()
        self.use_relu = use_relu
        
    def forward(self, x):
        return self.relu(self.conv_block(x)) if self.use_relu else self.conv_block(x)
    

class SeperableResidualConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SeperableResidualConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.first_block = SeperableConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3, use_relu=True)
        self.second_block = SeperableConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3, use_relu = False)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size = 1, stride = 2)

    def forward(self, x):
        residual = x
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.relu(torch.add(residual, x))
        return self.relu(self.conv(x))

        

class XceptionNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(XceptionNet, self).__init__()
        self.first_block = nn.Sequential(
            BasicConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=0),
            BasicConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
        )

        layers = list()
        last_layer_out = 64
        for size in [128, 256, 512, 728]:
            layers.append(SeperableResidualConvBlock(in_channels=last_layer_out, out_channels=size))
            last_layer_out = size
        layers.append(SeperableConvBlock(in_channels=last_layer_out, out_channels=1024, kernel_size=3, use_relu=True))
        layers.append(nn.AvgPool2d(kernel_size=3))
        self.sep_res_layers = nn.Sequential(*layers)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 1024)
        self.relu = nn.ReLU()

        if num_classes == 2:
            self.last_act = nn.Sigmoid()
            self.fc2 = nn.Linear(1024, 1)
        else:
            self.last_act = nn.Softmax(dim=1)
            self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.first_block(x)
        x = self.sep_res_layers(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.last_act(self.fc2(x))
        return x


def test():
    rand_tensor = torch.rand((1, 3, 256, 256))
    net = XceptionNet(3, 10)
    print(net(rand_tensor))


if __name__ == '__main__':
    test()
