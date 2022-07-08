#普通类型
import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision 
import torchvision.transforms as transforms

#初始图片大小为224

class ChannelAttention(nn.Module):
    def __init__(self,in_channel):
        super(ChannelAttention,self).__init__()
        self.maxpool = nn.MaxPool2d(1)
        self.avgpool = nn.AvgPool2d(1)
        #5//2=2（2.5向负无穷方向取整为2），同时-5//2=-3（-2.5向负无穷方向取整为-3）
        #经过pool之后  维度编程 channel * 1*1大小
        #对于channel*1*1来说 卷积 和 mlp相同
        self.fc1 = nn.Conv2d(in_channels=in_channel,out_channels=in_channel//16,kernel_size=1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channel//16,in_channel,kernel_size=1,bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self,t):
        max_out = self.maxpool(t)
        max_out = self.fc2(self.relu(self.fc1(max_out)))

        avg_out = self.avgpool(t)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))

        out = self.sigmod(max_out+avg_out)

        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,kernel_size=7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,t):
        #c*H*W
        max_out,_= torch.max(t,dim=1,keepdim=True)
        avg_out = torch.mean(t,dim=1,keepdim=True)

        out = torch.cat([max_out,avg_out],dim=1)
        out = self.sigmoid(self.conv1(out))

        return out

class BasicBlock(nn.Module):
    expansion=1
    def __init___(self,in_channel,out_channel,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel
                                ,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                                kernel_size=3,stride=1,padding=1,bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.channel_atten= ChannelAttention(self.expansion*out_channel)
        self.spatial_atten = SpatialAttention()
        self.downsample = downsample

    def forward(self,t):
        #t 输入
        identity = t
        if self.downsample is not None:
            identity = self.downsample(t)
        out = self.relu(self.bn1(self.conv1(t)))
        out = self.bn2(self.conv2(t))
        CBAM_Channel = out * self.channel_atten(out)
        out = CBAM_Channel*self.spatial_atten(CBAM_Channel)
        out = self.relu(out+identity)

        return out

class Bottlenect(nn.Module):
    expansion = 4

    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(Bottlenect,self).__init__()
        #conv1没有padding 因为 维持原型 只改变通道数 第一层中接到池化后面 所以相当于通道数没有改变
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                                kernel_size=3,stride=stride,padding = 1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel*self.expansion,
                                kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.channel_atten= ChannelAttention(self.expansion*out_channel)
        self.spatial_atten = SpatialAttention()
        self.downsample = downsample

    def forward(self,t):
        identity = t
        if self.downsample is not None:
            identity = self.downsample(t)
        
        out = self.relu(self.bn1(self.conv1(t)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        CBAM_Channel = out * self.channel_atten(out)
        out = CBAM_Channel*self.spatial_atten(CBAM_Channel)
        out = self.relu(out+identity)

        return out

class ResNet(nn.Module):
    def __init__(self,block,block_num,num_classes=1000):
        super(ResNet,self).__init__()
        self.in_channel=64
        #o=112 (224-7+2p)/2 +1 =112   so p = 3
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.in_channel,kernel_size=7,
                                stride=2,padding=3,bias=False)
        #o=56  (112-3+2p)/2+1 = 56 so p=1  在pytorch中所有计算结果都是向下取整
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block,64,block_num[0])
        self.layer2 = self._make_layer(block,128,block_num[1],stride=2)
        self.layer3 = self._make_layer(block,256,block_num[2],stride=2)
        self.layer4 = self._make_layer(block,512,block_num[3],stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_classes)

        #初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def _make_layer(self,block,channel,block_num,stride=1):
        downsample =None
        if stride != 1 or self.in_channel!=block.expansion * channel:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel,out_channels=block.expansion*channel,kernel_size=1,
                        stride=stride,bias=False),
                nn.BatchNorm2d(block.expansion*channel)
            )

        layers =[]
        layers.append(block(in_channel=self.in_channel,out_channel=channel,stride=stride,downsample=downsample))

        self.in_channel = block.expansion * channel

        for _ in range(1,block_num):
            layers.append(block(in_channel=self.in_channel,out_channel=channel))

        return nn.Sequential(*layers)

    def forward(self,t):
        out = self.relu(self.bn1(self.conv1(t)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)

        return out

def CB_resnet18(num_classes =1000):
    return ResNet(BasicBlock,[2,2,2,2],num_classes)

def CB_resent34(num_classes = 1000):
    return ResNet(BasicBlock,[3,4,6,3],num_classes)

def CB_resnet50(num_classes=1000):
    return ResNet(Bottlenect,[3,4,6,3],num_classes)

def CB_resnet101(num_classes=1000):
    return ResNet(Bottlenect,[3,4,23,3],num_classes)

def CB_resnet152(num_classes=1000):
    return ResNet(Bottlenect,[3,8,36,3],num_classes)