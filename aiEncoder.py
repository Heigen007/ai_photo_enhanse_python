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
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0001
DATASET_SIZE = 52000

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

# Трансформация для 128x128
transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.LANCZOS),
    transforms.ToTensor()
])

dataset = ImageDataset(IMAGE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Определение автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # **Encoder (U-Net)**
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1), # 8x8 -> 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        # **Decoder (U-Net with skip connections)**
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 256, kernel_size=3, stride=1, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        encoded = self.enc5(e4)

        # Decoder **WITH skip connections**
        d1 = self.dec1(encoded) + e4  # 8x8
        d2 = self.dec2(d1) + e3       # 16x16
        d3 = self.dec3(d2) + e2       # 32x32
        d4 = self.dec4(d3) + e1       # 64x64
        decoded = self.dec5(d4)       # 128x128

        return encoded, decoded

# Функция загрузки модели
def load_model():
    model = Autoencoder()
    if os.path.exists("autoencoder.pth"):
        print("Загружаем существующую модель...")
        model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
        print("Модель загружена.")
    return model

# Функция обучения
def train_model():
    model = load_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # **Dynamic Learning Rate**
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

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
            if batch_idx % 203 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Средний Loss: {avg_loss:.6f}")

        scheduler.step(avg_loss)
        print(f"Текущий Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

        if (epoch + 1) % 10 == 0:
            model_path = f"autoencoder_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Модель сохранена: {model_path}")

    torch.save(model.state_dict(), "autoencoder.pth")
    print("Финальная модель автоэнкодера сохранена!")

    end_time = time.time()
    print(f"Обучение завершено за {(end_time - start_time) / 60:.2f} минут.")

# Запуск
if __name__ == "__main__":
    train_model()



# Средний Loss: 0.0005 для v1 с 3 слоями
# Средний Loss: 0.0026 для v2 с 3 слоями
# Средний Loss: 0.0039 для v3 с 5 слоями
# Средний Loss: 0.0007 для 1 гипотезы
# Средний Loss: 0.0006 для 2 гипотезы