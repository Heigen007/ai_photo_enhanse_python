import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.utils as vutils

# Параметры
LOW_RES_DIR = "64x64/faces"
HIGH_RES_DIR = "128x128/faces"
OUTPUT_DIR = "output_models_srgan"
RESULTS_DIR = "results_srgan"
BATCH_SIZE = 16
EPOCHS = 30
LR_G = 0.0001
LR_D = 0.00005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Кастомный датасет
class SuperResDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform_low=None, transform_high=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.image_files = [f for f in os.listdir(low_res_dir) if f.endswith('.png')]
        self.transform_low = transform_low
        self.transform_high = transform_high

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        low_res_path = os.path.join(self.low_res_dir, img_name)
        high_res_path = os.path.join(self.high_res_dir, img_name)

        low_res_img = Image.open(low_res_path).convert("RGB")
        high_res_img = Image.open(high_res_path).convert("RGB")

        if self.transform_low:
            low_res_img = self.transform_low(low_res_img)
        if self.transform_high:
            high_res_img = self.transform_high(high_res_img)

        return low_res_img, high_res_img

# Трансформации (без нормализации, диапазон [0, 1])
low_res_transform = transforms.Compose([
    transforms.ToTensor()
])

high_res_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = SuperResDataset(LOW_RES_DIR, HIGH_RES_DIR, low_res_transform, high_res_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Тестовое изображение
test_low_res_img = Image.open(os.path.join(LOW_RES_DIR, "00000.png")).convert("RGB")
test_high_res_img = Image.open(os.path.join(HIGH_RES_DIR, "00000.png")).convert("RGB")
test_low_res_tensor = low_res_transform(test_low_res_img).unsqueeze(0).to(DEVICE)
test_high_res_tensor = high_res_transform(test_high_res_img).unsqueeze(0).to(DEVICE)

# Генератор
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[self._make_residual_block(64) for _ in range(5)]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Dropout(0.1)
        )

        self.final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        x = self.initial(x)
        res = self.res_blocks(x)
        x = self.mid(x + res)
        x = self.upsample(x)
        x = self.final(x)
        return torch.sigmoid(x)  # [0, 1]

# Дискриминатор
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

# VGG для перцептивной потери
class VGG19Loss(nn.Module):
    def __init__(self):
        super(VGG19Loss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg)[:35]).to(DEVICE).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        # VGG ожидает вход в [0, 1], нормализация не нужна, так как данные уже в [0, 1]
        return self.vgg(x)

# Функция загрузки моделей
def load_models():
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    g_path = os.path.join(OUTPUT_DIR, "generator.pth")
    d_path = os.path.join(OUTPUT_DIR, "discriminator.pth")
    if os.path.exists(g_path):
        print("Загружаем генератор...")
        generator.load_state_dict(torch.load(g_path, map_location=DEVICE))
    if os.path.exists(d_path):
        print("Загружаем дискриминатор...")
        discriminator.load_state_dict(torch.load(d_path, map_location=DEVICE))
    return generator, discriminator

# Функция оценки
def evaluate_model(generator, low_res_tensor, high_res_tensor):
    generator.eval()
    with torch.no_grad():
        output = generator(low_res_tensor)
        output = output.cpu().numpy()  # Уже в [0, 1]
        high_res = high_res_tensor.cpu().numpy()  # Уже в [0, 1]
        psnr_value = psnr(high_res[0].transpose(1, 2, 0), output[0].transpose(1, 2, 0), data_range=1.0)
        ssim_value = ssim(high_res[0].transpose(1, 2, 0), output[0].transpose(1, 2, 0), channel_axis=2, data_range=1.0)
    return psnr_value, ssim_value

# Функция обучения
def train_model():
    generator, discriminator = load_models()
    vgg_loss = VGG19Loss()

    g_optimizer = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.9, 0.999))
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    real_label = torch.ones((BATCH_SIZE, 1, 1, 1), device=DEVICE) * 0.9
    fake_label = torch.zeros((BATCH_SIZE, 1, 1, 1), device=DEVICE)

    start_time = time.time()
    print("Начинаем обучение SRGAN...")

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        g_loss_total, d_loss_total = 0, 0

        for batch_idx, (low_res, high_res) in enumerate(dataloader):
            low_res, high_res = low_res.to(DEVICE), high_res.to(DEVICE)

            # Обучение дискриминатора
            d_optimizer.zero_grad()
            real_output = discriminator(high_res)
            d_loss_real = bce_loss(real_output, real_label[:real_output.size(0)])

            fake_high_res = generator(low_res)
            fake_output = discriminator(fake_high_res.detach())
            d_loss_fake = bce_loss(fake_output, fake_label[:fake_output.size(0)])

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()
            d_loss_total += d_loss.item()

            # Обучение генератора
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_high_res)
            g_loss_adv = bce_loss(fake_output, real_label[:fake_output.size(0)])

            vgg_real = vgg_loss(high_res)
            vgg_fake = vgg_loss(fake_high_res)
            g_loss_content = mse_loss(vgg_fake, vgg_real)

            pixel_loss = mse_loss(fake_high_res, high_res)
            g_loss = g_loss_content + 0.01 * g_loss_adv + 0.1 * pixel_loss
            g_loss.backward()
            g_optimizer.step()
            g_loss_total += g_loss.item()

            if batch_idx % 35 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.6f}, G Loss: {g_loss.item():.6f}")

        avg_g_loss = g_loss_total / len(dataloader)
        avg_d_loss = d_loss_total / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] завершена, Avg D Loss: {avg_d_loss:.6f}, Avg G Loss: {avg_g_loss:.6f}")

        # Оценка PSNR и SSIM
        psnr_value, ssim_value = evaluate_model(generator, test_low_res_tensor, test_high_res_tensor)
        print(f"PSNR (00000.png): {psnr_value:.2f}, SSIM (00000.png): {ssim_value:.4f}")

        # Сохранение тестового изображения
        with torch.no_grad():
            test_output = generator(test_low_res_tensor)
            vutils.save_image(test_output, os.path.join(RESULTS_DIR, f"epoch_{epoch+1:02d}_00000.png"))

        if (epoch + 1) % 1 == 0:
            torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, f"discriminator_epoch_{epoch+1}.pth"))
            print(f"Модели сохранены: epoch {epoch+1}")

    torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, "discriminator.pth"))
    print("Финальные модели сохранены!")
    print(f"Обучение завершено за {(time.time() - start_time) / 60:.2f} минут.")

if __name__ == "__main__":
    train_model()