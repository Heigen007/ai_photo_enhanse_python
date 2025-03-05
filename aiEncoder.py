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
EPOCHS = 40
LEARNING_RATE = 0.0001
DATASET_SIZE = 52000

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

# Трансформация для 128x128
transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.LANCZOS),  # Убеждаемся, что вход 128x128
    transforms.ToTensor()
])

dataset = ImageDataset(IMAGE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Определение автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)   # 128x128 -> 64x64
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 64x64 -> 32x32
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 32x32 -> 16x16
        self.enc4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.enc5 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1) # 8x8 -> 8x8

        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 256, kernel_size=3, stride=1, padding=1)  
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(e1))
        e3 = torch.relu(self.enc3(e2))
        e4 = torch.relu(self.enc4(e3))
        encoded = torch.relu(self.enc5(e4))

        # Decoder with skip connections
        d1 = torch.relu(self.dec1(encoded) + e4) # 8x8
        d2 = torch.relu(self.dec2(d1) + e3)      # 16x16
        d3 = torch.relu(self.dec3(d2) + e2)      # 32x32
        d4 = torch.relu(self.dec4(d3) + e1)      # 64x64
        decoded = torch.sigmoid(self.dec5(d4))   # 128x128

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

            if batch_idx % 204 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] завершена, Средний Loss: {avg_loss:.4f}")

        # Шаг уменьшения скорости обучения
        scheduler.step(avg_loss)

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

# Средний Loss: 0.0005 для v1 с 3 слоями
# Средний Loss: 0.0026 для v2 с 3 слоями
# Средний Loss: 0.0039 для v3 с 5 слоями
# Средний Loss: 0.0007 для 1 гипотезы
# Обучаем модель(добавляем гладкое снижение learning rate + добавляем эпохи)
# Средний Loss: 0.0004 для 2 гипотезы