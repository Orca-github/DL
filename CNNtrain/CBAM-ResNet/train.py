import torch
import torchvision
import torch.nn as nn
from modelselfv2 import CB_resnet18
from modelselfv2 import CB_resent34
from modelselfv2 import CB_resnet50
from modelselfv2 import CB_resnet101
from modelselfv2 import CB_resnet152

import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

train_set = torchvision.datasets.CIFAR10(
    root='../data/CIFAR10/'
    ,train=True
    ,download=False
    ,transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
        
    ])
)

test_set = torchvision.datasets.CIFAR10(
    root='../data/CIFAR10/'
    ,train=False
    ,download=False
    ,transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
        
    ])
)
#34
# useCBAM = []
# useCBAM.append([1,0,0])
# useCBAM.append([1,0,0,0])
# useCBAM.append([1,0,0,0,0,0])
# useCBAM.append([1,0,0])
#50
# useCBAM = []
# useCBAM.append([1,0,0])
# useCBAM.append([1,0,0,0])
# useCBAM.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# useCBAM.append([1,0,0])
#101
# useCBAM = []
# useCBAM.append([1,0])
# useCBAM.append([1,0])
# useCBAM.append([1,0])
# useCBAM.append([1,0])
#152
# useCBAM = []
# useCBAM.append([1,0])
# useCBAM.append([1,0])
# useCBAM.append([1,0])
# useCBAM.append([1,0])
#---------------init netwrok shape
useCBAM = []
useCBAM.append([1,1])
useCBAM.append([1,1])
useCBAM.append([1,1])
useCBAM.append([1,1])

#------------------network
network =  CB_resnet18(num_classes=10,useCBAM=useCBAM).to('cuda')

#------------------prepare data
train_loader = torch.utils.data.DataLoader(train_set,batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=6)

optimizer = optim.Adam(network.parameters(),lr = 0.0001)

#----------------------train
total_loss = 0
total_correct = 0

train_counter = []
test_counter = []

train_losses =[]
test_losses=[]
for epoch in range(5):
    total_loss = 0
    total_correct = 0
    train_counter.append(epoch)
    test_counter.append(epoch)
    for batch in train_loader:
        images = batch[0].to('cuda')
        labels = batch[1].to('cuda')
        
        preds = network(images)
        loss = F.cross_entropy(preds,labels)
        #------------?????????????????? ??????pytorch???????????????
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
    train_losses.append(total_correct/len(train_set))
    print("epoch:",epoch,"train total_correct",total_correct,"train loss:",total_loss,"Accuracy :",total_correct/len(train_set))

    total_loss_eval = 0
    total_correct_eval = 0
    network.eval()  # ????????????????????????????????????????????????????????????model.eval()
    with torch.no_grad():  # torch.no_grad?????????tensor????????????x??????requires_grad = True??????x????????????tensor????????????w-?????????requires_grad??????False??????grad_fn??????None,????????????w??????
        for image, labels in test_loader: 
            image = image.to('cuda')
            labels = labels.to('cuda')
            preds = network(image)
            loss = F.cross_entropy(preds,labels)
            total_loss_eval += loss.item()  # ?????????????????????loss
        # ????????????????????????
            total_correct_eval += get_num_correct(preds,labels)  #  ?????????????????????
    total_loss_eval /= len(test_loader.dataset)
    test_losses.append(total_loss_eval)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_loss_eval, total_correct_eval, len(test_loader.dataset),
        100. * total_correct_eval / len(test_loader.dataset))) # print
print('train',total_correct/len(train_set))

#print
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')  # x??????train_counter?????? y??????train_losses?????? ????????????
plt.scatter(test_counter, test_losses, color='red')  # ???????????????
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  # ??????????????????
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

#----------------save weight
torch.save(network.state_dict(),'cb18pt')