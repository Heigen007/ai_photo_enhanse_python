import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Создаем папку для сжатых изображений
os.makedirs("compressed_images", exist_ok=True)

# Параметры
IMAGE_DIR = "images"  # Папка с изображениями
OUTPUT_DIR = "compressed_images"  # Где сохраняем результаты
RESULTS_FILE = "jpeg_webp_results.csv"  # Файл для сохранения результатов

# Функция вычисления метрик PSNR, SSIM и размера файла
def calculate_metrics(original, compressed):
    if not os.path.exists(original) or not os.path.exists(compressed):
        print(f"Ошибка: один из файлов не найден: {original} или {compressed}")
        return None, None, None, None
    
    original_img = cv2.imread(original)
    compressed_img = cv2.imread(compressed)
    
    if original_img is None or compressed_img is None:
        print(f"Ошибка: невозможно загрузить изображение {original} или {compressed}")
        return None, None, None, None
    
    psnr = cv2.PSNR(original_img, compressed_img)
    ssim_value = ssim(original_img, compressed_img, channel_axis=2, win_size=3)
    original_size = os.path.getsize(original) / 1024  # Размер в КБ
    compressed_size = os.path.getsize(compressed) / 1024  # Размер в КБ
    
    return psnr, ssim_value, original_size, compressed_size

# Функция сохранения изображения в разных форматах
def save_compressed_images(image_path):
    img = cv2.imread(image_path)
    filename = os.path.basename(image_path).split('.')[0]
    
    # JPEG 50% качество
    jpeg_path = os.path.join(OUTPUT_DIR, f"{filename}_jpeg.jpg")
    cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    
    # WebP 50% качество
    webp_path = os.path.join(OUTPUT_DIR, f"{filename}_webp.webp")
    cv2.imwrite(webp_path, img, [cv2.IMWRITE_WEBP_QUALITY, 50])
    
    return jpeg_path, webp_path

# Загружаем изображения и сжимаем
image_files = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith(('jpg', 'png'))]

results = []
for img_path in image_files:
    jpeg_path, webp_path = save_compressed_images(img_path)
    psnr_jpeg, ssim_jpeg, orig_size, jpeg_size = calculate_metrics(img_path, jpeg_path)
    psnr_webp, ssim_webp, _, webp_size = calculate_metrics(img_path, webp_path)
    
    if psnr_jpeg is not None and psnr_webp is not None:
        results.append([os.path.basename(img_path), "JPEG", psnr_jpeg, ssim_jpeg, orig_size, jpeg_size])
        results.append([os.path.basename(img_path), "WebP", psnr_webp, ssim_webp, orig_size, webp_size])

# Создание таблицы и сохранение в файл
results_df = pd.DataFrame(results, columns=["Image", "Format", "PSNR", "SSIM", "Original Size (KB)", "Compressed Size (KB)"])
results_df.to_csv(RESULTS_FILE, index=False)

print(f"Результаты сохранены в {RESULTS_FILE}")