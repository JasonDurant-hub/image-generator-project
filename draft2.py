import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
trainset=torchvision.datasets.CIFAR10(root="D:\ML 2026.07\data",train=True,transform=transform,download=True)
trainloader=torch.utils.data.DataLoader(dataset=trainset,batch_size=64,shuffle=True,num_workers=2)

class VAE(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim=latent_dim
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1) # 256*4*4
        self.bn3=nn.BatchNorm2d(256)
        self.relu=nn.ReLU()
        self.fc_mu=nn.Linear(256*4*4,self.latent_dim)
        self.fc_logvar=nn.Linear(256*4*4,self.latent_dim)
        self.dec_fc=nn.Linear(self.latent_dim,256*4*4)
        self.dec_conv1=nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1) 
        self.dec_conv2=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1)
        self.dec_conv3=nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,stride=2,padding=1) 
        self.sigmoid=nn.Sigmoid()
    def encoder(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        x=self.relu(self.bn3(self.conv3(x)))
        x=x.view(x.size(0),256*4*4)
        mu=self.fc_mu(x)
        logvar=self.fc_logvar(x)
        return mu,logvar
    def reparameterize(self,mu,logvar):
        eps=torch.randn_like(input=mu)
        std=torch.exp(0.5*logvar)
        return mu+eps*std
    def decoder(self,z):
        z=self.dec_fc(z)
        z=z.view(z.size(0),256,4,4)
        z=self.relu(self.bn2(self.dec_conv1(z)))
        z=self.relu(self.bn1(self.dec_conv2(z)))
        z=self.sigmoid(self.dec_conv3(z))
        return z
    def forward(self,x):
        mu,logvar=self.encoder(x)
        z=self.reparameterize(mu,logvar)
        out=self.decoder(z)
        return mu,logvar,out
    def loss(self,x,mu,logvar,z):
        self.recon_loss=nn.functional.mse_loss(x,z,reduction="sum")
        self.kl_loss=-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
        return (self.recon_loss+self.kl_loss)/x.size(0)
    
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")  
model=VAE(latent_dim=512).to(device)
optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-5)


epoch=10
for epoch in range(epoch):
    training_loss=0.0
    for data,_ in trainloader:
        data=data.to(device)
        optimizer.zero_grad()
        mu,logvar,out=model(data)
        vae_loss=model.loss(data,mu,logvar,out)
        vae_loss.backward()
        optimizer.step()
        training_loss+=vae_loss.item()
    print(f"epoch:{epoch+1}  Average loss: {training_loss / len(trainloader.dataset):.4f}")

    model.eval()
    z=torch.rand_like(64,512).to(device)
    out=model.decoder(z).cpu()
    out=out.view(64,3,32,)

z=torch.rand(32)


        


# batch*3*32*32  >> batch*256*4*4 cancat  >>  batch*512  >> mu logvar >> z >> loss  
# train
# test z=randn(32,3,32,32) labels=torch