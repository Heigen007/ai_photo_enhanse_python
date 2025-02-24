import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import sys

# Параметры
IMAGE_DIR = "unsplashPhotos"
ENCODED_DIR = "encoded_images"
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0003

# Создание папки для сжатых представлений
os.makedirs(ENCODED_DIR, exist_ok=True)

# Функция для приведения изображений к квадратному формату с сохранением пропорций
def pad_to_square(image, fill=0):
    width, height = image.size
    max_dim = max(width, height)
    new_image = Image.new("RGB", (max_dim, max_dim), (fill, fill, fill))
    new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
    return new_image

# Функция загрузки изображений
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
        orig_size = image.size  # Сохраняем оригинальный размер

        # Делаем изображение квадратным, сохраняя пропорции
        image = pad_to_square(image)

        if self.transform:
            image = self.transform(image)
        
        return image, img_path, orig_size  # Передаем оригинальный размер

# ✅ Исправленная трансформация
transform = transforms.Compose([
    transforms.Resize(256, interpolation=Image.LANCZOS),  # Изменяем размер, но пропорции уже сохранены
    transforms.ToTensor()
])

dataset = ImageDataset(IMAGE_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Определение автоэнкодера
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Функция загрузки модели
def load_model():
    model = Autoencoder()
    if os.path.exists("autoencoder.pth"):
        print("Загружаем существующую модель...")
        model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
        print("Модель загружена.")
    return model

# Основной запуск обучения
def train_model():
    model = load_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Начинаем обучение модели...")
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_idx, (images, _, _) in enumerate(dataloader):
            optimizer.zero_grad()
            encoded, outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Обновление статуса во время эпохи
            progress = (batch_idx + 1) / num_batches * 100
            sys.stdout.write(f"\rEpoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{num_batches}] | Progress: {progress:.2f}%")
            sys.stdout.flush()
        
        print(f" | Loss: {total_loss / len(dataloader):.4f}")
    
    # Сохранение модели
    torch.save(model.state_dict(), "autoencoder.pth")
    print("Модель автоэнкодера сохранена!")

if __name__ == "__main__":
    train_model()
