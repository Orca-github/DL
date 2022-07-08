import torch
import torch.nn as nn

#18layer 34layer 结构
class BasicBlock(nn.Module):
    expansion = 1#用于标注是否需要转换卷积核个数
    def __init__(self,in_channel,out_channel,stride =1,downsample=None):
        #downsample 下采样 在高层网络中的 进行降维（宽高，非通道）的那个操作 56，56，64--*stride：2---1，1，128*----28，28，128
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                kernel_size=3,stride=stride,padding=1,bias=False)
        #若 stride=1 paddi=1则 宽高不会改变 output= (input -3 +2*1)/1 + 1 = input
        #若 stride=2 padding=1则： output = (input -3 + 2*1) /2 +1
        # = input/2 +0.5
        # = input/2(向下取整)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                                kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    def forward(self,t):
        identity = t
        if self.downsample is not None:
            identity = self.downsample(t)
        t = self.conv1(t)
        t = self.bn1(t)
        t = self.relu(t)

        t = self.conv2(t)
        t = self.bn2(t)
        
        t += identity
        t = self.relu(t)

        return t


#50layer 101layer 152layer 结构
class Bottlenect(nn.Module):
    expansion = 4#用于标注是否需要转换卷积核个数
    def __init__(self,in_channel,out_channel,stride =1,downsample=None):
        #downsample 下采样 在高层网络中的 进行降维（宽高，非通道）的那个操作 56，56，64--*stride：2---1，1，128*----28，28，128
        super(Bottlenect,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                kernel_size=1,stride=1,padding=1,bias=False)
        #若 stride=1 paddi=1则 宽高不会改变 output= (input -3 +2*1)/1 + 1 = input
        #若 stride=2 padding=1则： output = (input -3 + 2*1) /2 +1
        # = input/2 +0.5
        # = input/2(向下取整)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        #----------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                                kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        #------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel*self.expansion,
                                kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(implace=True)
        self.downsample = downsample
    
    def forward(self,t):
        identity = t
        if self.downsample is not None:
            identity = self.downsample(t)
        t = self.conv1(t)
        t = self.bn1(t)
        t = self.relu(t)

        t = self.conv2(t)
        t = self.bn2(t)
        t = self.relu(t)

        t = self.conv3(t)
        t = self.bn3(t)
        
        t += identity
        t = self.relu(t)

        return t

class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes =1000,include_top=True):
        super(ResNet,self).__init__()
        self.include_top = include_top#是否包含全连接层 用于后期搭建更为复杂的网络
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,
                                padding=3,bias=False)#..nn.conv2d中第一个参数就是输入channel 第二个参数为输出的channel
        #输入channel3 表示 rgb彩色图像
        #O = (n-f+2p)/s  + 1      in-224 out-112  (224-7+2p)2 + 1 =112
        #p = 2 或 3  带回去算 向下取整 所以用 3 
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace = True)
        #pool后宽高为56*56
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #block就是 basicblock（不需要转换为度 18 layer &34 layer 或者 bottlenect（需要转换维度）
        self.layer1 = self._make_layer(block,64, blocks_num[0])
        self.layer2 = self._make_layer(block,128, blocks_num[1],stride = 2)
        self.layer3 = self._make_layer(block,256, blocks_num[2],stride = 2)
        self.layer4 = self._make_layer(block,512, blocks_num[3],stride =2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode= 'fan_out', nonlinearity='relu')


    def _make_layer(self,block, channel, block_num,stride=1):
        downsample = None

        #18 和 34层会跳过下 if语句   self.in_channel =64 若是需要改变维度启用bottleneck则bottle中的expansion为4 
        #layer2 中没有传入参数 stride stirde默认为1
        if stride!=1 or self.in_channel != channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,channel * block.expansion, kernel_size=1,
                stride = stride, bias= False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel,channel,downsample=downsample,stride=stride))
        self.in_channel = channel*block.expansion# 50 101 152的每个block的第三层的channel数量为1 2 层的4倍

        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel))
        
        return nn.Sequential(*layers)

    def forward(self,t):
        t = self.conv1(t)
        t = self.bn1(t)
        t = self.relu(t)
        t = self.maxpool(t)

        t = self.layer1(t)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.layer4(t)

        if self.include_top:
            t = self.avgpool(t)
            t = torch.flatten(t,1)
            t = self.fc(t)
        
        return t

def resnet18 (num_classes=1000,include_top = True):
    return ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes,include_top=include_top)


def resnet34 (num_classes=1000,include_top = True):
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)

def resnet50 (num_classes=1000,include_top = True):
    return ResNet(Bottlenect,[3,4,6,3],num_classes=num_classes,include_top=include_top)


def resnet101 (num_classes=1000,include_top = True):
    return ResNet(Bottlenect,[3,4,23,3],num_classes=num_classes,include_top=include_top)
