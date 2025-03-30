import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Параметры
IMAGE_DIR = "images"
ENCODED_DIR = "encoded_images"
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

os.makedirs(ENCODED_DIR, exist_ok=True)

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Трансформация для 1024x1024
transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.LANCZOS),
    transforms.ToTensor()
])

dataset = ImageDataset(IMAGE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Улучшенная архитектура автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )  # 1024x1024 -> 512x512

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )  # 512x512 -> 256x256

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )  # 256x256 -> 128x128

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )  # 128x128 -> 64x64

        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )  # 64x64 -> 32x32

        self.enc6 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )  # 32x32 -> 32x32 (bottleneck)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )  # 32x32 -> 32x32

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )  # 32x32 -> 64x64

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )  # 64x64 -> 128x128

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )  # 128x128 -> 256x256

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )  # 256x256 -> 512x512

        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )  # 512x512 -> 1024x1024

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        encoded = self.enc6(e5)

        # Decoder (с Skip Connections)
        d1 = self.dec1(encoded) + e5
        d2 = self.dec2(d1) + e4
        d3 = self.dec3(d2) + e3
        d4 = self.dec4(d3) + e2
        d5 = self.dec5(d4) + e1
        decoded = self.dec6(d5)

        return encoded, decoded

def load_model():
    model = Autoencoder()
    if os.path.exists("autoencoder.pth"):
        print("Загружаем существующую модель...")
        model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
        print("Модель загружена.")
    return model

def train_model():
    model = load_model()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    start_time = time.time()
    print("Начинаем обучение модели...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()
            encoded, outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.8f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] завершена, Средний Loss: {avg_loss:.8f}")

        scheduler.step(avg_loss)

        if (epoch + 1) % 2 == 0:
            model_path = f"autoencoder_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Модель сохранена: {model_path}")

    torch.save(model.state_dict(), "autoencoder.pth")
    print("Финальная модель автоэнкодера сохранена!")

    print(f"Обучение завершено за {(time.time() - start_time) / 60:.2f} минут.")

if __name__ == "__main__":
    train_model()

# Средний Loss: 0.0005 для v1 с 3 слоями
# Средний Loss: 0.0026 для v2 с 3 слоями
# Средний Loss: 0.0039 для v3 с 5 слоями
# Средний Loss: 0.0007 для 1 гипотезы
# Обучаем модель(добавляем гладкое снижение learning rate + добавляем эпохи)
# Средний Loss: 0.0004 для 2 гипотезы

# 1024x1024

# 10 эпох - 0.0023 - 530 минут
# +5 эпох - 0.0020 - 166 минут
#+30 эпох - 0.0007 - 1115 минут
# Итого: 1811 минута или 30 часов - 0.0007

# Добавляем LeakyReLU
# 14 эпох + 2 эпохи + 10 эпох + 16 эпох + 12 эпох (54) - 0.00038