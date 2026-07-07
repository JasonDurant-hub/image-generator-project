import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
trainset=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(batch_size=64,shuffle=True)
testset=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(batch_size=64,batch_size=64,shuffle=False)

class CNN(nn.modules):
    def __init__(self):
        super.__init__()
        self.conv1=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self

    