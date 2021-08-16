import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d

# Bottleneck block 만들기
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size= 1, stride= 1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride= stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size= 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential() # short_cut을 위한 identity 만들기
        
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels , kernel_size= 1, stride= stride),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self , x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)        
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)

        return x

# Residual 블락 만들기
class Residual_block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * Residual_block.expansion, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels * Residual_block.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != Residual_block.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Residual_block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Residual_block.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


# ResNet 만들기 
class ResNet(nn.Module): # [3, 4, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__() # 논문에 나온 residual block전 해야 할 것들을 생성
        self.in_channels = 64 
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size= 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride = 2, padding = 1)

        # Resnet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels= 64, stride= 1)
        self.layer2 = self._make_layer(block, layers[1], out_channels= 128, stride= 2)
        self.layer3 = self._make_layer(block, layers[2], out_channels= 256, stride= 2)
        self.layer4 = self._make_layer(block, layers[3], out_channels= 512, stride= 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion , num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_blocks, out_channels, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


def ResNet18(img_channels = 3, num_classes = 1000):
    return ResNet(Residual_block, [2, 2, 2, 2], img_channels, num_classes)


def ResNet34(img_channels = 3, num_classes = 1000):
    return ResNet(Residual_block, [3, 4, 6, 3], img_channels, num_classes)


def ResNet50(img_channels = 3, num_classes = 1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels = 3, num_classes = 1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels = 3, num_classes = 1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], img_channels, num_classes)

def test():
    net1 = ResNet34()
    x = torch.randn(2, 3, 224, 224)
    y1 = net1(x).to('cpu')
    print(y1.shape)

    net2 = ResNet152()
    y2 = net2(x).to('cpu')
    print(y2.shape)

test()