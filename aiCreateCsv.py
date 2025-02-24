# generate_csv.py
import os
import cv2
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import gzip

# Параметры
IMAGE_DIR = "images"
OUTPUT_DIR = "compressed_images"
ENCODED_DIR = "encoded_images"
RESULTS_FILE = "ai_results.csv"

# Создание папки для сжатых изображений
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ENCODED_DIR, exist_ok=True)  # Добавляем, если папка не существует

dataset = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('jpg', 'png'))]

def load_model():
    from aiEncoder import Autoencoder  # Импортируем из файла обучения
    model = Autoencoder()
    model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Загружаем модель без запуска обучения
if not os.path.exists("autoencoder.pth"):
    raise FileNotFoundError("Модель autoencoder.pth не найдена. Пожалуйста, сначала обучите модель, запустив train_model.py")

model = load_model()

# ✅ **Исправленный transform**
transform = transforms.Compose([
    transforms.Resize(256, interpolation=Image.LANCZOS),  # Пропорциональное изменение
    transforms.ToTensor()
])

results = []

with torch.no_grad():
    for img_name in dataset:
        img_path = os.path.join(IMAGE_DIR, img_name)
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size  # Сохраняем оригинальный размер

        input_tensor = transform(image).unsqueeze(0)
        encoded, decoded = model(input_tensor)

        # ✅ **Сохранение сжатого представления**
        encoded_path = os.path.join(ENCODED_DIR, f"{img_name}.pt")
        torch.save(encoded.cpu().half(), gzip.open(encoded_path + ".gz", "wb"))

        # **Измеряем размер сжатого файла**
        compressed_size = os.path.getsize(encoded_path + ".gz") / 1024  # Учитываем сжатый файл

        # **Восстановление изображения (разжатие)**
        decoded_image = transforms.ToPILImage()(decoded.squeeze().cpu().clamp(0, 1))
        decoded_image = decoded_image.resize(orig_size, Image.LANCZOS)  # Восстанавливаем размеры
        decoded_image_path = os.path.join(OUTPUT_DIR, f"{img_name}_decoded.png")
        decoded_image.save(decoded_image_path)

        # **Сравнение с оригиналом**
        original_image = cv2.imread(img_path)
        decoded_image_cv = cv2.imread(decoded_image_path)

        if original_image.shape[:2] != decoded_image_cv.shape[:2]:
            decoded_image_cv = cv2.resize(decoded_image_cv, (original_image.shape[1], original_image.shape[0]))

        psnr = cv2.PSNR(original_image, decoded_image_cv)
        ssim_value = ssim(original_image, decoded_image_cv, channel_axis=2, win_size=3)
        original_size = os.path.getsize(img_path) / 1024  # Размер оригинала в КБ

        results.append([img_name, "AI", psnr, ssim_value, original_size, compressed_size])

# **Создание CSV**
results_df = pd.DataFrame(results, columns=["Image", "Format", "PSNR", "SSIM", "Original Size (KB)", "Compressed Size (KB)"])
results_df.to_csv(RESULTS_FILE, index=False)

print(f"Результаты сохранены в {RESULTS_FILE}")
