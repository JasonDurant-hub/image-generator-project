import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform_train = transforms.Compose([
    transforms.RandomCrop(size=32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])
transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)   #32*32*16
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)   #32*32*32
        self.bn2=nn.BatchNorm2d(32) 
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)  #16*16*32
        self.conv3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)  #16*16*64
        self.bn3=nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1) #16*16*128
        self.bn4=nn.BatchNorm2d(128)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)   #8*8*128
        self.conv5=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)  #8*8*256
        self.bn5=nn.BatchNorm2d(256)
        self.conv6=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)  #8*8*512
        self.bn6=nn.BatchNorm2d(512)
        self.pool3=nn.MaxPool2d(stride=2,kernel_size=2)  #4*4*512
 
        self.fc1=nn.Linear(4*4*512,256)
        self.bn7=nn.BatchNorm1d(256)
        self.fc2=nn.Linear(256,10)
        self.relu=nn.ReLU()
        
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))  
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.pool1(x)  
    
        x=self.relu(self.bn3(self.conv3(x)))
        x=self.relu(self.bn4(self.conv4(x)))
        x=self.pool2(x)

        x=self.relu(self.bn5(self.conv5(x)))
        x=self.relu(self.bn6(self.conv6(x)))
        x=self.pool3(x)

        x=x.view(-1,4*4*512)
        x=self.fc1(x)
        x=self.bn7(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x


def main():
    model=SimpleCNN()

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    trainset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True,num_workers=2)
    testset=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform_test)
    testloader=torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False,num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    model.to(device)
    epochs=30
    for epoch in range(epochs):
        running_loss=0.0
        for i,data in enumerate(trainloader):
            inputs,labels=data[0].to(device),data[1].to(device)
            optimizer.zero_grad()
            outputs=model.forward(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i%200==199:
                print(f"[Epoch:{epoch+1} batch:{i+1}] loss:{running_loss/200:.3f}")
                running_loss=0.0
    print('Finished Training!')

    correct=0
    total=0
    with torch.no_grad():
        for data in testloader:
            inputs,labels=data[0].to(device),data[1].to(device)
            outputs=model(inputs)
            _,predicted=torch.max(outputs,1)
            total+=labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'模型在测试集的准确率: {100 * correct / total} %')

if __name__ == '__main__':
    main()

"""
优化部分 1.数据的预处理 增加训练量
        2.加深网络+(bn)
        3.optimizer
"""