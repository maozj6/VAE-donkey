import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import os

# 设定参数
image_size = (120, 160)
batch_size = 64
latent_dim = 256  # 增加 latent_dim
epochs = 50
lr = 5e-4  # 降低学习率，防止过拟合
save_dir = "vae_results_road"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

data_dir = "./mini_carson/trainA/"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# VAE 模型
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128 * 15 * 20, latent_dim)
        self.fc_logvar = nn.Linear(128 * 15 * 20, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 15 * 20)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.fc_decode(z).view(-1, 128, 15, 20)
        x_recon = self.decoder(x_recon)
        return x_recon, mu, logvar

vae = VAE(latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=lr)

# KL Annealing
for epoch in range(epochs):
    kl_weight = min(1.0, epoch / 100)  # KL 退火
    vae.train()
    total_loss = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon_imgs, mu, logvar = vae(imgs)

        recon_loss = F.l1_loss(recon_imgs, imgs)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_weight * kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # **保存原图和重建图像**
    vae.eval()
    with torch.no_grad():
        sample_imgs, _ = next(iter(dataloader))
        sample_imgs = sample_imgs[:8].to(device)  # 取8张图片
        recon_imgs, _, _ = vae(sample_imgs)

        comparison = torch.cat([sample_imgs.cpu(), recon_imgs.cpu()], dim=0)
        vutils.save_image(comparison, f"{save_dir}/epoch_{epoch + 1}.png", nrow=8)

# **保存 VAE 模型**
torch.save(vae.state_dict(), f"{save_dir}/vae_b.pth")
print("训练完成，模型已保存！")
