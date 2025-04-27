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
LOW_RES_DIR = "128x128/faces"
HIGH_RES_DIR = "256x256/faces"
OUTPUT_DIR = "output_models_srgan_128to256"
RESULTS_DIR = "results_srgan_128to256"
BATCH_SIZE = 12
EPOCHS = 50
LR_G = 0.00005  # Оставляем как есть
LR_D = 0.00001  # Оставляем как есть
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

# Трансформации (диапазон [0, 1])
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

# Генератор (с 12 residual-блоками, двухшаговый upsampling: 128x128 -> 192x192 -> 256x256)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[self._make_residual_block(64) for _ in range(12)]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        # Одношаговый upsampling (128x128 -> 256x256)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 128x128 -> 256x256
            nn.PReLU(),
            nn.Dropout(0.3)
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
        return torch.sigmoid(x)

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
        noise = torch.randn_like(x) * 0.01
        x = x + noise
        return self.net(x)

# VGG для перцептивной потери (обрезаем до conv4_4, слой 20)
class VGG19Loss(nn.Module):
    def __init__(self):
        super(VGG19Loss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg = nn.Sequential(*list(vgg)[:20]).to(DEVICE).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.normalize(x)
        return self.vgg(x)

# Функция для цветовой потери (HSV)
def color_loss(fake, real):
    mse_loss = nn.MSELoss()
    
    def rgb_to_hsv_torch(rgb):
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        c_max, _ = torch.max(rgb, dim=1)
        c_min, _ = torch.min(rgb, dim=1)
        delta = c_max - c_min
        
        h = torch.zeros_like(c_max)
        mask = (c_max == r) & (delta != 0)
        h[mask] = 60 * (((g[mask] - b[mask]) / delta[mask]) % 6)
        mask = (c_max == g) & (delta != 0)
        h[mask] = 60 * ((b[mask] - r[mask]) / delta[mask] + 2)
        mask = (c_max == b) & (delta != 0)
        h[mask] = 60 * ((r[mask] - g[mask]) / delta[mask] + 4)
        h = h / 360
        
        s = torch.zeros_like(c_max)
        mask = c_max != 0
        s[mask] = delta[mask] / c_max[mask]
        
        v = c_max
        
        return torch.stack([h, s, v], dim=1)

    fake_hsv = rgb_to_hsv_torch(fake)
    real_hsv = rgb_to_hsv_torch(real)
    return mse_loss(fake_hsv[:, 0, :, :], real_hsv[:, 0, :, :])

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
        output = output.cpu().numpy()
        high_res = high_res_tensor.cpu().numpy()
        psnr_value = psnr(high_res[0].transpose(1, 2, 0), output[0].transpose(1, 2, 0), data_range=1.0)
        ssim_value = ssim(high_res[0].transpose(1, 2, 0), output[0].transpose(1, 2, 0), channel_axis=2, data_range=1.0)
    return psnr_value, ssim_value

# Функция обучения
def train_model():
    generator, discriminator = load_models()
    vgg_loss = VGG19Loss()

    g_optimizer = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.9, 0.999), weight_decay=1e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.9, 0.999))
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    real_label = torch.ones((BATCH_SIZE, 1, 1, 1), device=DEVICE) * 0.8
    fake_label = torch.zeros((BATCH_SIZE, 1, 1, 1), device=DEVICE) + 0.2

    start_time = time.time()
    print("Начинаем обучение SRGAN...")

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        g_loss_total, d_loss_total = 0, 0
        g_loss_content_total, g_loss_adv_total, pixel_loss_total, color_loss_total = 0, 0, 0, 0

        for batch_idx, (low_res, high_res) in enumerate(dataloader):
            low_res, high_res = low_res.to(DEVICE), high_res.to(DEVICE)

            # Проверка размеров входных данных (128x128 -> 256x256)
            if low_res.shape[2:] != (128, 128) or high_res.shape[2:] != (256, 256):
                print(f"Неверный размер: low_res {low_res.shape}, high_res {high_res.shape}")
                continue

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
            color_loss_value = color_loss(fake_high_res, high_res)
            
            # Считаем итоговую потерю (веса оставляем как есть)
            g_loss = 0.1 * g_loss_content + 0.05 * g_loss_adv + 2.0 * pixel_loss + 0.7 * color_loss_value
            g_loss.backward()
            g_optimizer.step()
            
            # Суммируем компоненты потерь для логирования
            g_loss_total += g_loss.item()
            g_loss_content_total += g_loss_content.item()
            g_loss_adv_total += g_loss_adv.item()
            pixel_loss_total += pixel_loss.item()
            color_loss_total += color_loss_value.item()

            if batch_idx % 45 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.6f}, G Loss: {g_loss.item():.6f}, "
                      f"G Content: {g_loss_content.item():.6f}, G Adv: {g_loss_adv.item():.6f}, "
                      f"Pixel: {pixel_loss.item():.6f}, Color: {color_loss_value.item():.6f}")

        avg_g_loss = g_loss_total / len(dataloader)
        avg_d_loss = d_loss_total / len(dataloader)
        avg_g_loss_content = g_loss_content_total / len(dataloader)
        avg_g_loss_adv = g_loss_adv_total / len(dataloader)
        avg_pixel_loss = pixel_loss_total / len(dataloader)
        avg_color_loss = color_loss_total / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] завершена, Avg D Loss: {avg_d_loss:.6f}, Avg G Loss: {avg_g_loss:.6f}")
        print(f"Avg G Content: {avg_g_loss_content:.6f}, Avg G Adv: {avg_g_loss_adv:.6f}, "
              f"Avg Pixel: {avg_pixel_loss:.6f}, Avg Color: {avg_color_loss:.6f}")

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

# v0 - PSNR (00000.png): 8.21, SSIM (00000.png): 0.0422
# v1 - PSNR (00000.png): 10.44, SSIM (00000.png): 0.0566
# v2 - PSNR (00000.png): 26.43, SSIM (00000.png): 0.8029
# v3 - переходим на фото 128 -> 256, потому что на фото 64x64 очень мало деталей, поэтому дорисовывать не из чего - и так нет изначального понятия, что дорисовывать
# v3 - PSNR (00000.png): 30.18, SSIM (00000.png): 0.8573
# v4 - PSNR (00000.png): 29.74, SSIM (00000.png): 0.8557

# Начинаем обучение SRGAN...
# Epoch [1/50], Batch [0/84], D Loss: 0.693636, G Loss: 17.526901, G Content: 170.830063, G Adv: 0.677015, Pixel: 0.086926, Color: 0.337417
# Epoch [1/50], Batch [45/84], D Loss: 0.658779, G Loss: 9.719891, G Content: 93.908600, G Adv: 0.758105, Pixel: 0.037676, Color: 0.308247
# Epoch [1/50] завершена, Avg D Loss: 0.653787, Avg G Loss: 10.233411
# Avg G Content: 98.691145, Avg G Adv: 0.745274, Avg Pixel: 0.047079, Avg Color: 0.332678
# PSNR (00000.png): 18.18, SSIM (00000.png): 0.5996
# Модели сохранены: epoch 1
# Epoch [2/50], Batch [0/84], D Loss: 0.623311, G Loss: 6.755611, G Content: 64.295341, G Adv: 0.801653, Pixel: 0.017319, Color: 0.359082
# Epoch [2/50], Batch [45/84], D Loss: 0.587578, G Loss: 6.964721, G Content: 66.242218, G Adv: 0.885418, Pixel: 0.013815, Color: 0.383711
# Epoch [2/50] завершена, Avg D Loss: 0.580598, Avg G Loss: 6.635624
# Avg G Content: 62.942478, Avg G Adv: 0.898659, Avg Pixel: 0.016012, Avg Color: 0.377740
# PSNR (00000.png): 20.28, SSIM (00000.png): 0.6938
# Модели сохранены: epoch 2
# Epoch [3/50], Batch [0/84], D Loss: 0.559824, G Loss: 5.522217, G Content: 51.616779, G Adv: 0.963720, Pixel: 0.011439, Color: 0.413537
# Epoch [3/50], Batch [45/84], D Loss: 0.521495, G Loss: 5.637918, G Content: 52.910583, G Adv: 1.091579, Pixel: 0.012441, Color: 0.381997
# Epoch [3/50] завершена, Avg D Loss: 0.533931, Avg G Loss: 5.592652
# Avg G Content: 52.429494, Avg G Adv: 1.057324, Avg Pixel: 0.011814, Avg Color: 0.390298
# PSNR (00000.png): 21.25, SSIM (00000.png): 0.7276
# Модели сохранены: epoch 3
# Epoch [4/50], Batch [0/84], D Loss: 0.526078, G Loss: 5.848394, G Content: 54.958157, G Adv: 1.150754, Pixel: 0.010551, Color: 0.391341
# Epoch [4/50], Batch [45/84], D Loss: 0.517463, G Loss: 5.262924, G Content: 49.121319, G Adv: 1.171898, Pixel: 0.010951, Color: 0.386135
# Epoch [4/50] завершена, Avg D Loss: 0.515777, Avg G Loss: 4.980593
# Avg G Content: 46.330617, Avg G Adv: 1.178417, Avg Pixel: 0.009529, Avg Color: 0.385073
# PSNR (00000.png): 21.51, SSIM (00000.png): 0.7430
# Модели сохранены: epoch 4
# Epoch [5/50], Batch [0/84], D Loss: 0.518141, G Loss: 4.484663, G Content: 41.583706, G Adv: 1.216957, Pixel: 0.006480, Color: 0.360692
# Epoch [5/50], Batch [45/84], D Loss: 0.507642, G Loss: 4.266726, G Content: 39.167999, G Adv: 1.267507, Pixel: 0.008101, Color: 0.386212
# Epoch [5/50] завершена, Avg D Loss: 0.508189, Avg G Loss: 4.577546
# Avg G Content: 42.371292, Avg G Adv: 1.252731, Avg Pixel: 0.008625, Avg Color: 0.372187
# PSNR (00000.png): 22.86, SSIM (00000.png): 0.7501
# Модели сохранены: epoch 5
# Epoch [6/50], Batch [0/84], D Loss: 0.507959, G Loss: 4.619750, G Content: 42.958286, G Adv: 1.181863, Pixel: 0.006304, Color: 0.360317
# Epoch [6/50], Batch [45/84], D Loss: 0.518851, G Loss: 4.827977, G Content: 45.324200, G Adv: 1.167009, Pixel: 0.006268, Color: 0.320957
# Epoch [6/50] завершена, Avg D Loss: 0.520444, Avg G Loss: 4.240238
# Avg G Content: 39.262223, Avg G Adv: 1.229586, Avg Pixel: 0.007158, Avg Color: 0.340317
# PSNR (00000.png): 24.28, SSIM (00000.png): 0.7691
# Модели сохранены: epoch 6
# Epoch [7/50], Batch [0/84], D Loss: 0.583830, G Loss: 4.357522, G Content: 41.087162, G Adv: 1.253317, Pixel: 0.004883, Color: 0.251961
# Epoch [7/50], Batch [45/84], D Loss: 0.503855, G Loss: 3.870990, G Content: 35.723270, G Adv: 1.350100, Pixel: 0.005607, Color: 0.314206
# Epoch [7/50] завершена, Avg D Loss: 0.519104, Avg G Loss: 3.905839
# Avg G Content: 36.380853, Avg G Adv: 1.211979, Avg Pixel: 0.005621, Avg Color: 0.279876
# PSNR (00000.png): 25.40, SSIM (00000.png): 0.7918
# Модели сохранены: epoch 7
# Epoch [8/50], Batch [0/84], D Loss: 0.603932, G Loss: 3.598404, G Content: 33.849743, G Adv: 0.844219, Pixel: 0.003754, Color: 0.233873
# Epoch [8/50], Batch [45/84], D Loss: 0.721428, G Loss: 3.392710, G Content: 31.924774, G Adv: 0.618282, Pixel: 0.004438, Color: 0.229203
# Epoch [8/50] завершена, Avg D Loss: 0.563097, Avg G Loss: 3.656291
# Avg G Content: 34.373770, Avg G Adv: 1.100524, Avg Pixel: 0.004760, Avg Color: 0.220526
# PSNR (00000.png): 26.16, SSIM (00000.png): 0.7959
# Модели сохранены: epoch 8
# Epoch [9/50], Batch [0/84], D Loss: 0.604584, G Loss: 3.237943, G Content: 30.426414, G Adv: 0.723117, Pixel: 0.003707, Color: 0.216760
# Epoch [9/50], Batch [45/84], D Loss: 0.577575, G Loss: 3.940844, G Content: 37.434063, G Adv: 1.069872, Pixel: 0.003272, Color: 0.196286
# Epoch [9/50] завершена, Avg D Loss: 0.601976, Avg G Loss: 3.481768
# Avg G Content: 33.004740, Avg G Adv: 0.941928, Avg Pixel: 0.004334, Avg Color: 0.179329
# PSNR (00000.png): 27.56, SSIM (00000.png): 0.8109
# Модели сохранены: epoch 9
# Epoch [10/50], Batch [0/84], D Loss: 0.530784, G Loss: 3.994127, G Content: 38.205719, G Adv: 1.025937, Pixel: 0.003994, Color: 0.163244
# Epoch [10/50], Batch [45/84], D Loss: 0.535345, G Loss: 3.380347, G Content: 31.939133, G Adv: 1.136119, Pixel: 0.003336, Color: 0.175652
# Epoch [10/50] завершена, Avg D Loss: 0.587497, Avg G Loss: 3.334588
# Avg G Content: 31.616584, Avg G Adv: 0.980889, Avg Pixel: 0.004062, Avg Color: 0.165372
# PSNR (00000.png): 25.72, SSIM (00000.png): 0.8179
# Модели сохранены: epoch 10
# Epoch [11/50], Batch [0/84], D Loss: 0.563971, G Loss: 3.317779, G Content: 31.660479, G Adv: 0.982225, Pixel: 0.004302, Color: 0.134309
# Epoch [11/50], Batch [45/84], D Loss: 0.513771, G Loss: 3.205821, G Content: 30.064829, G Adv: 1.062575, Pixel: 0.002603, Color: 0.201433
# Epoch [11/50] завершена, Avg D Loss: 0.586975, Avg G Loss: 3.232234
# Avg G Content: 30.599221, Avg G Adv: 1.025104, Avg Pixel: 0.004031, Avg Color: 0.161419
# PSNR (00000.png): 28.01, SSIM (00000.png): 0.8238
# Модели сохранены: epoch 11
# Epoch [12/50], Batch [0/84], D Loss: 0.753589, G Loss: 3.507150, G Content: 33.284657, G Adv: 1.169213, Pixel: 0.007143, Color: 0.151338
# Epoch [12/50], Batch [45/84], D Loss: 0.568659, G Loss: 2.741771, G Content: 26.029970, G Adv: 0.849723, Pixel: 0.002368, Color: 0.130788
# Epoch [12/50] завершена, Avg D Loss: 0.589796, Avg G Loss: 3.106330
# Avg G Content: 29.469143, Avg G Adv: 0.955561, Avg Pixel: 0.003739, Avg Color: 0.148801
# PSNR (00000.png): 28.26, SSIM (00000.png): 0.8339
# Модели сохранены: epoch 12
# Epoch [13/50], Batch [0/84], D Loss: 0.601472, G Loss: 2.783358, G Content: 26.494789, G Adv: 0.742849, Pixel: 0.003351, Color: 0.128622
# Epoch [13/50], Batch [45/84], D Loss: 0.558658, G Loss: 2.944422, G Content: 27.690912, G Adv: 1.228729, Pixel: 0.005691, Color: 0.146446
# Epoch [13/50] завершена, Avg D Loss: 0.579459, Avg G Loss: 3.018772
# Avg G Content: 28.629519, Avg G Adv: 0.991760, Avg Pixel: 0.003897, Avg Color: 0.140626
# PSNR (00000.png): 28.55, SSIM (00000.png): 0.8364
# Модели сохранены: epoch 13
# Epoch [14/50], Batch [0/84], D Loss: 0.544134, G Loss: 3.202612, G Content: 30.643969, G Adv: 0.952701, Pixel: 0.003181, Color: 0.120311
# Epoch [14/50], Batch [45/84], D Loss: 0.506057, G Loss: 3.423963, G Content: 32.702274, G Adv: 1.250418, Pixel: 0.002954, Color: 0.121868
# Epoch [14/50] завершена, Avg D Loss: 0.580300, Avg G Loss: 2.941542
# Avg G Content: 27.958287, Avg G Adv: 0.995723, Avg Pixel: 0.003447, Avg Color: 0.127191
# PSNR (00000.png): 28.75, SSIM (00000.png): 0.8391
# Модели сохранены: epoch 14
# Epoch [15/50], Batch [0/84], D Loss: 0.691816, G Loss: 2.893217, G Content: 27.448660, G Adv: 0.738217, Pixel: 0.008730, Color: 0.134257
# Epoch [15/50], Batch [45/84], D Loss: 0.650488, G Loss: 2.830728, G Content: 26.916733, G Adv: 1.243085, Pixel: 0.002032, Color: 0.104050
# Epoch [15/50] завершена, Avg D Loss: 0.562104, Avg G Loss: 2.876996
# Avg G Content: 27.292208, Avg G Adv: 1.044841, Avg Pixel: 0.003440, Avg Color: 0.126647
# PSNR (00000.png): 29.04, SSIM (00000.png): 0.8450
# Модели сохранены: epoch 15


# // Тут я еачал дообучение, то есть 1 эпоха это 16

# Начинаем обучение SRGAN...
# Epoch [1/50], Batch [0/84], D Loss: 0.514015, G Loss: 3.219961, G Content: 30.568291, G Adv: 1.294584, Pixel: 0.002936, Color: 0.132186
# Epoch [1/50], Batch [45/84], D Loss: 0.524129, G Loss: 2.605242, G Content: 24.681633, G Adv: 1.098701, Pixel: 0.002542, Color: 0.110085
# Epoch [1/50] завершена, Avg D Loss: 0.575884, Avg G Loss: 2.821848
# Avg G Content: 26.785651, Avg G Adv: 1.015267, Avg Pixel: 0.003421, Avg Color: 0.122397
# PSNR (00000.png): 28.58, SSIM (00000.png): 0.8444
# Модели сохранены: epoch 1
# Epoch [2/50], Batch [0/84], D Loss: 0.722703, G Loss: 2.562007, G Content: 24.483788, G Adv: 0.602230, Pixel: 0.002787, Color: 0.111348
# Epoch [2/50], Batch [45/84], D Loss: 0.757549, G Loss: 2.665039, G Content: 25.132294, G Adv: 1.297431, Pixel: 0.002342, Color: 0.117506
# Epoch [2/50] завершена, Avg D Loss: 0.570899, Avg G Loss: 2.748131
# Avg G Content: 26.083956, Avg G Adv: 1.035528, Avg Pixel: 0.003327, Avg Color: 0.116150
# PSNR (00000.png): 28.99, SSIM (00000.png): 0.8486
# Модели сохранены: epoch 2
# Epoch [3/50], Batch [0/84], D Loss: 0.505806, G Loss: 2.311912, G Content: 21.601130, G Adv: 1.250167, Pixel: 0.001719, Color: 0.122646
# Epoch [3/50], Batch [45/84], D Loss: 0.585262, G Loss: 2.655411, G Content: 25.296875, G Adv: 0.799284, Pixel: 0.002131, Color: 0.116425
# Epoch [3/50] завершена, Avg D Loss: 0.559736, Avg G Loss: 2.710705
# Avg G Content: 25.673203, Avg G Adv: 1.066305, Avg Pixel: 0.003143, Avg Color: 0.119689
# PSNR (00000.png): 28.25, SSIM (00000.png): 0.8478
# Модели сохранены: epoch 3
# Epoch [4/50], Batch [0/84], D Loss: 0.534823, G Loss: 2.521182, G Content: 23.896187, G Adv: 0.835291, Pixel: 0.002094, Color: 0.122301
# Epoch [4/50], Batch [45/84], D Loss: 0.564785, G Loss: 2.246342, G Content: 20.966158, G Adv: 0.737750, Pixel: 0.002499, Color: 0.154058
# Epoch [4/50] завершена, Avg D Loss: 0.604624, Avg G Loss: 2.660522
# Avg G Content: 25.190368, Avg G Adv: 1.013272, Avg Pixel: 0.003131, Avg Color: 0.120800
# PSNR (00000.png): 29.21, SSIM (00000.png): 0.8453
# Модели сохранены: epoch 4
# Epoch [5/50], Batch [0/84], D Loss: 0.518686, G Loss: 2.612652, G Content: 24.482092, G Adv: 1.088397, Pixel: 0.002306, Color: 0.150588
# Epoch [5/50], Batch [45/84], D Loss: 0.564072, G Loss: 2.477931, G Content: 23.584753, G Adv: 0.768192, Pixel: 0.001963, Color: 0.110170
# Epoch [5/50] завершена, Avg D Loss: 0.556649, Avg G Loss: 2.628938
# Avg G Content: 24.848185, Avg G Adv: 1.035874, Avg Pixel: 0.003236, Avg Color: 0.122648
# PSNR (00000.png): 29.53, SSIM (00000.png): 0.8513
# Модели сохранены: epoch 5
# Epoch [6/50], Batch [0/84], D Loss: 0.518290, G Loss: 2.687692, G Content: 25.362150, G Adv: 1.103040, Pixel: 0.002417, Color: 0.130701
# Epoch [6/50], Batch [45/84], D Loss: 0.527048, G Loss: 2.673843, G Content: 25.444216, G Adv: 1.069024, Pixel: 0.001989, Color: 0.102846
# Epoch [6/50] завершена, Avg D Loss: 0.549917, Avg G Loss: 2.580459
# Avg G Content: 24.402496, Avg G Adv: 1.085773, Avg Pixel: 0.003019, Avg Color: 0.114119
# PSNR (00000.png): 29.39, SSIM (00000.png): 0.8517
# Модели сохранены: epoch 6
# Epoch [7/50], Batch [0/84], D Loss: 0.575116, G Loss: 3.043497, G Content: 29.140089, G Adv: 0.824086, Pixel: 0.003370, Color: 0.116492
# Epoch [7/50], Batch [45/84], D Loss: 0.604480, G Loss: 2.913429, G Content: 28.041548, G Adv: 0.662611, Pixel: 0.002211, Color: 0.102459
# Epoch [7/50] завершена, Avg D Loss: 0.563671, Avg G Loss: 2.556813
# Avg G Content: 24.161623, Avg G Adv: 1.069715, Avg Pixel: 0.003002, Avg Color: 0.115944
# PSNR (00000.png): 29.02, SSIM (00000.png): 0.8523
# Модели сохранены: epoch 7
# Epoch [8/50], Batch [0/84], D Loss: 0.575666, G Loss: 2.483696, G Content: 23.496832, G Adv: 1.160421, Pixel: 0.003831, Color: 0.097613
# Epoch [8/50], Batch [45/84], D Loss: 0.522837, G Loss: 2.712124, G Content: 25.674353, G Adv: 1.158401, Pixel: 0.001996, Color: 0.118253
# Epoch [8/50] завершена, Avg D Loss: 0.566314, Avg G Loss: 2.533400
# Avg G Content: 23.940656, Avg G Adv: 1.069123, Avg Pixel: 0.003153, Avg Color: 0.113675
# PSNR (00000.png): 29.08, SSIM (00000.png): 0.8497
# Модели сохранены: epoch 8
# Epoch [9/50], Batch [0/84], D Loss: 0.510355, G Loss: 2.855969, G Content: 27.182505, G Adv: 1.120532, Pixel: 0.001961, Color: 0.111098
# Epoch [9/50], Batch [45/84], D Loss: 0.554084, G Loss: 2.105999, G Content: 19.649836, G Adv: 1.045612, Pixel: 0.003254, Color: 0.117469
# Epoch [9/50] завершена, Avg D Loss: 0.569482, Avg G Loss: 2.503606
# Avg G Content: 23.621259, Avg G Adv: 1.076628, Avg Pixel: 0.003056, Avg Color: 0.116483
# PSNR (00000.png): 29.74, SSIM (00000.png): 0.8557
# Модели сохранены: epoch 9