import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pytorch_msssim  # SSIM Loss

# Параметры
IMAGE_DIR = "testImages"
ENCODED_DIR = "encoded_images"
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.001
DATASET_SIZE = 30

os.makedirs(ENCODED_DIR, exist_ok=True)

# Датасет
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

# Трансформация для 128x128 с нормализацией в [-1, 1]
transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация в [-1, 1]
])

dataset = ImageDataset(IMAGE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Определение автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(16),  # Добавлен BatchNorm
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),  # Добавлен Dropout

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64 -> 128x128
            nn.Tanh()  # Используем Tanh вместо Sigmoid
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def load_model():
    model = Autoencoder()
    if os.path.exists("autoencoder.pth"):
        print("Загружаем существующую модель...")
        model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
        print("Модель загружена.")
    return model

# Функция для обратного преобразования (из [-1,1] обратно в [0,1])
def denormalize(img):
    return (img + 1) / 2

def train_model():
    model = load_model()
    criterion = pytorch_msssim.SSIM(data_range=2.0, size_average=True, channel=3)  # SSIM Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    start_time = time.time()
    print("Начинаем обучение модели...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()
            encoded, outputs = model(images)
            loss = criterion(outputs, images)  # Используем SSIM Loss
            loss = 1 - loss  # SSIM даёт значение близкое к 1 для похожих картинок, поэтому инвертируем
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 542 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] завершена, Средний Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            model_path = f"autoencoder_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Модель сохранена: {model_path}")

    torch.save(model.state_dict(), "autoencoder.pth")
    print("Финальная модель автоэнкодера сохранена!")
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Обучение завершено за {elapsed_time:.2f} минут.")

if __name__ == "__main__":
    train_model()

# Средний Loss: 0.0005 для v1
# Средний Loss: 0.0026 для v2