import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

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

# 加载模型参数
latent_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim).to(device)
save_dir = "vae_results_road"
model_path = "./vae_results_road/vae_a.pth"

if os.path.exists(model_path):
    vae.load_state_dict(torch.load(model_path, map_location=device))
    print("模型加载成功！")
else:
    print("未找到模型文件！")
    exit()

vae.eval()

# 随机生成 latent vector 并重建图像
with torch.no_grad():
    random_z = torch.randn(1, 3,120,160).to(device)  # 随机生成 8 个 latent vector
    decoded_imgs = vae(random_z)

    print()
    # # 保存生成的图像
    # output_dir = f"{save_dir}/generated_images"
    # os.makedirs(output_dir, exist_ok=True)
    # vutils.save_image(decoded_imgs.cpu(), f"{output_dir}/random_reconstruction.png", nrow=8, normalize=True)
    # print(f"生成的图像已保存到 {output_dir}/random_reconstruction.png")
