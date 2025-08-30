import os
import av
from PIL import Image

# Папки с изображениями
AVIF_DIR = "avifImages"
IMAGE_DIR = "images"

# Убедимся, что папка назначения существует
os.makedirs(IMAGE_DIR, exist_ok=True)

# Функция конвертации AVIF в PNG
def convert_avif_to_png(image_path):
    """ Конвертирует AVIF в PNG через FFmpeg (pyav) """
    try:
        container = av.open(image_path)
        frame = next(container.decode(video=0))  # Декодируем первый кадр
        img = frame.to_ndarray(format="rgb24")  # Преобразуем в RGB

        filename = os.path.basename(image_path).split('.')[0] + ".png"
        png_path = os.path.join(IMAGE_DIR, filename)

        # Сохраняем файл в папку images
        Image.fromarray(img).save(png_path, format="PNG")

        print(f"Конвертировано: {image_path} -> {png_path}")
        return png_path
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None

# Функция обработки всех AVIF файлов
def process_avif_images():
    """ Находит все AVIF файлы в папке и конвертирует их в PNG """
    image_files = []
    for f in os.listdir(AVIF_DIR):
        if f.endswith('.avif'):
            png_path = convert_avif_to_png(os.path.join(AVIF_DIR, f))
            if png_path:
                image_files.append(png_path)
    return image_files

if __name__ == "__main__":
    converted_files = process_avif_images()
    print(f"Конвертированы файлы: {converted_files}")
