import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision 
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 6,out_channels=12,kernel_size=5)
        
        self.fc1=nn.Linear(in_features = 12*4*4, out_features = 12)
        self.fc2=nn.Linear(in_features = 12, out_features = 60)
        self.out=nn.Linear(in_features = 60, out_features = 10)
        
    def forward(self,t):
        t=t

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2, stride = 2)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=2, stride = 2)
        
        t = t.reshape(-1,12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
            
        
        return t
    
train_set = torchvision.datasets.FashionMNIST(
    root='./data/'
    ,train=True
    ,download=True
    ,transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

sample = next(iter(train_set))
image,label = sample

network = Network()
output = network(image.unsqueeze(0))
print(output)