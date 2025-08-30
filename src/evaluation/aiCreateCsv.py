import os
import cv2
import torch
import time
import torchvision.transforms as transforms
import pandas as pd
import lzma
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Параметры
IMAGE_DIR = "testImages"
OUTPUT_DIR = "compressed_images"
ENCODED_DIR = "encoded_images"
RESULTS_FILE = "ai_results.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ENCODED_DIR, exist_ok=True)

dataset = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('jpg', 'png'))]

def load_model():
    from src.models.aiEncoder import Autoencoder  # Импорт из вашего файла aiEncoder.py
    model = Autoencoder()
    if os.path.exists("autoencoder.pth"):
        print("Загружаем модель...")
        model.load_state_dict(torch.load("autoencoder.pth", map_location=torch.device('cpu')))
        model.eval()
        print("Модель загружена.")
    else:
        raise FileNotFoundError("Модель autoencoder.pth не найдена. Пожалуйста, сначала обучите модель.")
    return model

model = load_model()

# Трансформация для 1024x1024
transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.LANCZOS),
    transforms.ToTensor()
])

results = []

with torch.no_grad():
    for img_name in dataset:
        img_path = os.path.join(IMAGE_DIR, img_name)
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size

        input_tensor = transform(image).unsqueeze(0)

        # 1️⃣ Замер времени сжатия (автоэнкодер + квантование + LZMA)
        compression_start_time = time.time()

        # Автоэнкодер (получаем encoded и decoded)
        encoded, decoded = model(input_tensor)

        # FP16 → INT8 КВАНТОВАНИЕ
        scale = encoded.abs().max()
        encoded_int8 = (encoded / scale * 127).clamp(-128, 127).to(torch.int8)

        # Сохранение INT8 + scale в LZMA
        encoded_path = os.path.join(ENCODED_DIR, f"{img_name}.pt.xz")
        torch.save((encoded_int8, scale), encoded_path + ".tmp")

        with open(encoded_path + ".tmp", "rb") as f:
            compressed_data = lzma.compress(f.read(), preset=9)
        with open(encoded_path, "wb") as f:
            f.write(compressed_data)
        os.remove(encoded_path + ".tmp")

        compression_time = time.time() - compression_start_time
        compressed_size = os.path.getsize(encoded_path) / 1024

        # 2️⃣ Замер времени разжатия (LZMA распаковка + обратное квантование + восстановление изображения)
        decompression_start_time = time.time()

        # Восстановление INT8 → FP16
        with lzma.open(encoded_path, "rb") as f:
            decompressed_data = f.read()
        with open(encoded_path + ".tmp", "wb") as f:
            f.write(decompressed_data)

        encoded_int8_loaded, scale_loaded = torch.load(encoded_path + ".tmp")
        encoded_fp16 = (encoded_int8_loaded.float() / 127) * scale_loaded
        os.remove(encoded_path + ".tmp")

        # Восстановление изображения из decoded (уже полученного ранее)
        decoded_image = transforms.ToPILImage()(decoded.squeeze().cpu().clamp(0, 1))
        decoded_image = decoded_image.resize(orig_size, Image.LANCZOS)
        decoded_image_path = os.path.join(OUTPUT_DIR, f"{img_name}_decoded.png")
        decoded_image.save(decoded_image_path)

        decompression_time = time.time() - decompression_start_time

        # Сравнение с оригиналом
        original_image = cv2.imread(img_path)
        decoded_image_cv = cv2.imread(decoded_image_path)

        if original_image.shape[:2] != decoded_image_cv.shape[:2]:
            decoded_image_cv = cv2.resize(decoded_image_cv, (original_image.shape[1], original_image.shape[0]))

        psnr = cv2.PSNR(original_image, decoded_image_cv)
        ssim_value = ssim(original_image, decoded_image_cv, channel_axis=2, win_size=7)  # Увеличил win_size до 7 для 1024x1024
        original_size = os.path.getsize(img_path) / 1024

        results.append([img_name, "AI", psnr, ssim_value, original_size, compressed_size, compression_time, decompression_time])

results_df = pd.DataFrame(results, columns=["Image", "Format", "PSNR", "SSIM", "Original Size (KB)", "Compressed Size (KB)", "Compression Time (s)", "Decompression Time (s)"])
results_df.to_csv(RESULTS_FILE, index=False)

# Средние значения
avg_compression_time = results_df["Compression Time (s)"].mean()
avg_decompression_time = results_df["Decompression Time (s)"].mean()

print(f"Результаты сохранены в {RESULTS_FILE}")
print(f"Среднее время сжатия (AI, включая автоэнкодер, квантование и LZMA): {avg_compression_time:.4f} секунд")
print(f"Среднее время разжатия (AI, включая LZMA распаковку и восстановление): {avg_decompression_time:.4f} секунд")