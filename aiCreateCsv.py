import os
import cv2
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import gzip

# Параметры
IMAGE_DIR = "testImages"
OUTPUT_DIR = "compressed_images"
ENCODED_DIR = "encoded_images"
RESULTS_FILE = "ai_results.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ENCODED_DIR, exist_ok=True)

dataset = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('jpg', 'png'))]

def load_model():
    from aiEncoder import Autoencoder  # Импортируем из первого файла
    model = Autoencoder()
    model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

if not os.path.exists("autoencoder.pth"):
    raise FileNotFoundError("Модель autoencoder.pth не найдена. Пожалуйста, сначала обучите модель.")

model = load_model()

transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.LANCZOS),  # Убеждаемся, что вход 128x128
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

        # Сохранение сжатого представления с gzip
        encoded_half = encoded.cpu().half()  # float16
        encoded_path = os.path.join(ENCODED_DIR, f"{img_name}.pt.gz")
        torch.save(encoded_half, encoded_path + ".tmp")  # Временный файл
        with open(encoded_path + ".tmp", "rb") as f:
            compressed_data = gzip.compress(f.read())
        with open(encoded_path, "wb") as f:
            f.write(compressed_data)
        os.remove(encoded_path + ".tmp")

        compressed_size = os.path.getsize(encoded_path) / 1024  # Размер в КБ

        # Восстановление изображения
        decoded_image = transforms.ToPILImage()(decoded.squeeze().cpu().clamp(0, 1))
        decoded_image = decoded_image.resize(orig_size, Image.LANCZOS)
        decoded_image_path = os.path.join(OUTPUT_DIR, f"{img_name}_decoded.png")
        decoded_image.save(decoded_image_path)

        # Сравнение с оригиналом
        original_image = cv2.imread(img_path)
        decoded_image_cv = cv2.imread(decoded_image_path)

        if original_image.shape[:2] != decoded_image_cv.shape[:2]:
            decoded_image_cv = cv2.resize(decoded_image_cv, (original_image.shape[1], original_image.shape[0]))

        psnr = cv2.PSNR(original_image, decoded_image_cv)
        ssim_value = ssim(original_image, decoded_image_cv, channel_axis=2, win_size=3)
        original_size = os.path.getsize(img_path) / 1024

        results.append([img_name, "AI", psnr, ssim_value, original_size, compressed_size])

results_df = pd.DataFrame(results, columns=["Image", "Format", "PSNR", "SSIM", "Original Size (KB)", "Compressed Size (KB)"])
results_df.to_csv(RESULTS_FILE, index=False)

print(f"Результаты сохранены в {RESULTS_FILE}")