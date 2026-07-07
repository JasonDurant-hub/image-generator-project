import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
from torchvision.utils import save_image


def main():

    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])   #像素值在0到1之间
    batch_size=64
    trainset=torchvision.datasets.CIFAR10(root="D:/ML 2026.07/data",train=True,transform=transform,download=True)
    trainloader=torch.utils.data.DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True,num_workers=2)

    class VAE(nn.Module):
        def __init__(self,latent_dim):
            super().__init__()
            self.latent_dim=latent_dim
            self.enc_conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1)  # 3*32*32 >> 64*16*16
            self.bn1=nn.BatchNorm2d(64)
            self.enc_conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1) # 64*16*16 >> 128*8*8
            self.bn2=nn.BatchNorm2d(128)
            self.enc_conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1)  # 128*8*8 >> 256*4*4
            self.bn3=nn.BatchNorm2d(256)
            self.fc_mu=nn.Linear(256*4*4,self.latent_dim)  
            self.fc_logvar=nn.Linear(256*4*4,self.latent_dim)  
            self.dec_fc=nn.Linear(self.latent_dim,256*4*4)
            self.dec_conv1=nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1) #256*4*4 >> 128*8*8
            self.dec_bn1=nn.BatchNorm2d(128)
            self.dec_conv2=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1) #128*8*8 >> 64*16*16
            self.dec_bn2=nn.BatchNorm2d(64)
            self.dec_conv3=nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,stride=2,padding=1) # 64*16*16 >> 3*32*32
        def encoder(self,x):
            x=torch.relu(self.bn1(self.enc_conv1(x)))
            x=torch.relu(self.bn2(self.enc_conv2(x)))
            x=torch.relu(self.bn3(self.enc_conv3(x)))
            x=x.view(x.size(0),-1)
            mu=self.fc_mu(x)
            logvar=self.fc_logvar(x)
            return mu,logvar
        def reparameterize(self,mu,logvar):
            var=torch.exp(0.5*logvar)
            eps=torch.randn_like(input=mu)
            return mu+var*eps
        def decoder(self,z):
            z=self.dec_fc(z)
            z=z.view(z.size(0),256,4,4)
            z=torch.relu(self.dec_bn1(self.dec_conv1(z)))
            z=torch.relu(self.dec_bn2(self.dec_conv2(z)))
            z=self.dec_conv3(z)
            z=torch.sigmoid(z)
            return z
        def forward(self,x):
            mu,logvar=self.encoder(x)
            z=self.reparameterize(mu=mu,logvar=logvar)
            out=self.decoder(z)
            return out,mu,logvar

    def loss(x,recon_x,mu,logvar):
        loss_recon=nn.functional.mse_loss(x,recon_x,reduction="sum")
        loss_kl=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
        return (loss_recon+loss_kl)/x.size(0)

    device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
    model=VAE(latent_dim=512).to(device)
    optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


    os.makedirs(name="D:/ML 2026.07/result2",exist_ok=True)

    epoch=100
    train_loss=0.0
    for epoch in range(epoch):
        model.train()
        for data,_ in trainloader:
            data=data.to(device)
            optimizer.zero_grad()
            recon_x,mu,logvar=model(data)
            vae_loss=loss(data,recon_x,mu,logvar)
            vae_loss.backward()
            train_loss+=vae_loss.item()
            optimizer.step()
        print(f"epoch:{epoch+1}  Average loss: {train_loss / len(trainloader.dataset):.4f}")
        train_loss=0.0

        model.eval()
        with torch.no_grad():
            sample_z=torch.randn(32,512).to(device)
            sample_image=model.decoder(sample_z).cpu()
            save_image(sample_image.view(32,3,32,32),f"D:/ML 2026.07/result2/sample_epoch_{epoch+1}.jpg")

    print("Finished Training")

if __name__ == '__main__':
    main()


