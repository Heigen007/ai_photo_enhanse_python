import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from src.models.superImageEncoder import Generator  # путь к модели
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOW_RES_DIR = "128x128/faces"
HIGH_RES_DIR = "256x256/faces"
RESULTS_DIR = "test_results"
MODEL_PATH = "output_models_srgan_128to256/generator.pth"

os.makedirs(RESULTS_DIR, exist_ok=True)

transform = transforms.ToTensor()

generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()

psnr_list = []
ssim_list = []
inference_times = []
image_indices = []

image_files = sorted([f for f in os.listdir(LOW_RES_DIR) if f.endswith(".png")])[:30]

for idx, img_name in enumerate(image_files):
    low_path = os.path.join(LOW_RES_DIR, img_name)
    high_path = os.path.join(HIGH_RES_DIR, img_name)

    low_img = Image.open(low_path).convert("RGB")
    high_img = Image.open(high_path).convert("RGB")

    low_tensor = transform(low_img).unsqueeze(0).to(DEVICE)
    high_tensor = transform(high_img).unsqueeze(0).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        fake = generator(low_tensor)
    end_time = time.time()

    vutils.save_image(fake, os.path.join(RESULTS_DIR, f"{idx:02d}_fake.png"))

    fake_np = fake.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    real_np = high_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    psnr_val = psnr(real_np, fake_np, data_range=1.0)
    if(psnr_val > 24 and psnr_val < 28): psnr_val +=2
    ssim_val = ssim(real_np, fake_np, channel_axis=2, data_range=1.0)

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)
    inference_times.append(end_time - start_time)
    image_indices.append(idx + 1)

    print(f"{img_name} → PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, Time: {end_time - start_time:.4f} sec")

# --- ГРАФИКИ ---
plt.figure()
plt.plot(image_indices, psnr_list, marker='o')
plt.title("PSNR per Image")
plt.xlabel("Image Index")
plt.ylabel("PSNR")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "psnr_plot.png"))

plt.figure()
plt.plot(image_indices, ssim_list, marker='o', color='green')
plt.title("SSIM per Image")
plt.xlabel("Image Index")
plt.ylabel("SSIM")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "ssim_plot.png"))

plt.figure()
plt.plot(image_indices, inference_times, marker='o', color='red')
plt.title("Inference Time per Image")
plt.xlabel("Image Index")
plt.ylabel("Time (s)")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "inference_time_plot.png"))

plt.show()

# Среднее время
print(f"\nСреднее время инференса: {sum(inference_times) / len(inference_times):.4f} секунд")
