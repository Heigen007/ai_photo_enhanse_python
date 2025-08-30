import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from src.models.superImageEncoder import Generator  # Подключи свою модель

# Настройки
LOW_RES_DIR = "128x128/faces"
RESULTS_DIR = "test_results"
MODEL_PATH = "output_models_srgan_128to256/generator.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Загрузка модели
generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()

transform = transforms.ToTensor()

# Индексы изображений
for idx in [2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
    file = f"{idx:05d}.png"
    img = Image.open(os.path.join(LOW_RES_DIR, file)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = generator(tensor)

    vutils.save_image(output, os.path.join(RESULTS_DIR, f"enhanced_{idx:05d}.png"))
    print(f"Сохранено: enhanced_{idx:05d}.png")