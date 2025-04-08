import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
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
OUTPUT_DIR = "output_models"
RESULTS_DIR = "results"
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0004
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

# Трансформации
low_res_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

high_res_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = SuperResDataset(LOW_RES_DIR, HIGH_RES_DIR, low_res_transform, high_res_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Тестовое изображение (00000.png)
test_low_res_img = Image.open(os.path.join(LOW_RES_DIR, "00000.png")).convert("RGB")
test_high_res_img = Image.open(os.path.join(HIGH_RES_DIR, "00000.png")).convert("RGB")
test_low_res_tensor = low_res_transform(test_low_res_img).unsqueeze(0).to(DEVICE)
test_high_res_tensor = high_res_transform(test_high_res_img).unsqueeze(0).to(DEVICE)

# Архитектура автоэнкодера
class SuperResolutionAutoencoder(nn.Module):
    def __init__(self):
        super(SuperResolutionAutoencoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )  # 64x64 -> 32x32

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )  # 32x32 -> 32x32

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )  # 32x32 -> 64x64

        self.skip_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )  # 64x64 -> 128x128

    def forward(self, x):
        e1 = self.enc1(x)
        encoded = self.enc2(e1)
        d1 = self.dec1(encoded)
        e1_upsampled = self.skip_upsample(e1)
        decoded = self.dec2(d1 + e1_upsampled)
        return encoded, decoded

# Функция загрузки модели
def load_model():
    model = SuperResolutionAutoencoder().to(DEVICE)
    model_path = os.path.join(OUTPUT_DIR, "superres_autoencoder.pth")
    if os.path.exists(model_path):
        print("Загружаем существующую модель...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Модель загружена.")
    return model

# Функция оценки (только для 00000.png)
def evaluate_model(model, low_res_tensor, high_res_tensor):
    model.eval()
    with torch.no_grad():
        _, output = model(low_res_tensor)
        output = output.cpu().numpy() * 0.5 + 0.5  # Де-нормализация
        high_res = high_res_tensor.cpu().numpy() * 0.5 + 0.5
        psnr_value = psnr(high_res[0].transpose(1, 2, 0), output[0].transpose(1, 2, 0), data_range=1.0)
        ssim_value = ssim(high_res[0].transpose(1, 2, 0), output[0].transpose(1, 2, 0), channel_axis=2, data_range=1.0)
    return psnr_value, ssim_value

# Функция обучения
def train_model():
    model = load_model()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    start_time = time.time()
    print("Начинаем обучение модели...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (low_res, high_res) in enumerate(dataloader):
            low_res, high_res = low_res.to(DEVICE), high_res.to(DEVICE)

            optimizer.zero_grad()
            _, outputs = model(low_res)
            loss = criterion(outputs, high_res)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 300 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.8f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] завершена, Средний Loss: {avg_loss:.8f}")

        # Оценка PSNR и SSIM для 00000.png
        avg_psnr, avg_ssim = evaluate_model(model, test_low_res_tensor, test_high_res_tensor)
        print(f"PSNR (00000.png): {avg_psnr:.2f}, SSIM (00000.png): {avg_ssim:.4f}")

        # Сохранение тестового изображения
        model.eval()
        with torch.no_grad():
            _, test_output = model(test_low_res_tensor)
            test_output = test_output * 0.5 + 0.5
            vutils.save_image(test_output, os.path.join(RESULTS_DIR, f"epoch_{epoch+1:02d}_00000.png"))

        scheduler.step(avg_loss)

        if (epoch + 1) % 2 == 0:
            model_path = os.path.join(OUTPUT_DIR, f"superres_autoencoder_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Модель сохранена: {model_path}")

    final_model_path = os.path.join(OUTPUT_DIR, "superres_autoencoder.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Финальная модель сохранена!")

    print(f"Обучение завершено за {(time.time() - start_time) / 60:.2f} минут.")

if __name__ == "__main__":
    train_model()