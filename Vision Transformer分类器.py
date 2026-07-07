import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform_train=transforms.Compose([
    transforms.RandomCrop(size=32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])
transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])

def main():

    trainset=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform_train)
    trainloader=torch.utils.data.DataLoader(trainset,shuffle=True,batch_size=64,num_workers=4)
    testset=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform_test)
    testloader=torch.utils.data.DataLoader(testset,shuffle=False,batch_size=64,num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class SimpleViT(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.patch_embed1=nn.Conv2d(in_channels=3,out_channels=128,kernel_size=2,stride=2)
            self.bn1=nn.BatchNorm2d(128)
            self.patch_embed2=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=2,stride=2)
            self.bn2=nn.BatchNorm2d(256)
            self.cls_token=nn.Parameter(torch.randn(1,1,256)) #初始是任意值
            self.pos_embed=nn.Parameter(torch.randn(1,65,256))  #初始是任意值
            encoder_layer=nn.TransformerEncoderLayer(d_model=256,nhead=8,dim_feedforward=512,batch_first=True)
            self.transformer=nn.TransformerEncoder(encoder_layer,num_layers=8)
            self.fc=nn.Linear(256,10)
            self.relu=nn.ReLU()
        def forward(self,x):
            x=self.relu(self.bn1(self.patch_embed1(x)))  # batch*128*16*16
            x=self.relu(self.bn2(self.patch_embed2(x)))  # batch*256*8*8
            x=x.flatten(2).transpose(1,2)  #batch*64*256
            cls_token=self.cls_token.expand(x.size(0),-1,-1)  
            x=torch.concat((x,cls_token),dim=1)   #batch*65*256
            x=x+self.pos_embed
            x=self.transformer(x)   #batch*65*256
            x=x[:,0,:]
            x=self.fc(x)
            return x

    model=SimpleViT()

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)



    device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    model.to(device)
    epoch=10
    for epoch in range(epoch):
        running_loss=0.0
        for i,data in enumerate(trainloader):
            optimizer.zero_grad()
            inputs,labels=data[0].to(device),data[1].to(device)
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if i%200==199:
                print(f"[epoch:{epoch+1} batch:{i+1}]  loss:{running_loss/200:.3f}")
                running_loss=0.0
    print('Finished Training!')

    total,correct=0,0
    for testdata in testloader:
        inputs,labels=testdata[0].to(device),testdata[1].to(device)
        outputs=model(inputs)
        _,predicted=torch.max(outputs,dim=1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print(f"Acurracy:{100*correct/total:.3f}%")

if __name__=="__main__":
    main()


"""优化的部分 1.optimizer从Adam变成了SGD (momentum,)
             2.刚开始做了一层>>两层conv,提高模型深度

"""
