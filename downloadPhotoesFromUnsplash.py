import os
import pandas as pd
import requests
from tqdm import tqdm

# Параметры
file_path = "unsplashDataset/photos.tsv000"  # Укажи путь к файлу
save_dir = "unsplashPhotos"  # Папка для фото
progress_file = "download_progress.txt"  # Файл с ID последнего загруженного фото
start_photo_id = None  # Укажи photo_id, с которого продолжить загрузку, или None, чтобы начать с начала

# Создание папки для сохранения фото
os.makedirs(save_dir, exist_ok=True)

# Читаем TSV-файл
df = pd.read_csv(file_path, sep="\t", usecols=["photo_id", "photo_image_url"])

# Если start_photo_id указан, находим, с какого места продолжить
if start_photo_id:
    if start_photo_id in df["photo_id"].values:
        start_index = df[df["photo_id"] == start_photo_id].index[0]
        df = df.iloc[start_index:]
    else:
        print(f"Ошибка: Указанный photo_id '{start_photo_id}' не найден в файле.")
        exit(1)

# Если start_photo_id не указан, но есть файл прогресса, продолжаем с него
elif os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        last_downloaded_id = f.read().strip()
    if last_downloaded_id in df["photo_id"].values:
        start_index = df[df["photo_id"] == last_downloaded_id].index[0] + 1
        df = df.iloc[start_index:]

# Загружаем фотографии по одной
for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
    photo_id = row["photo_id"]
    url = row["photo_image_url"].strip()  # Убираем пробелы
    if not url.startswith("http"):
        print(f"Пропуск {photo_id}, так как некорректный URL: {url}")
        continue

    file_extension = url.split("?")[0].split(".")[-1]  # Определяем расширение
    if len(file_extension) > 4:  # Иногда расширение длиннее 4 символов — значит, что-то не так
        file_extension = "jpg"

    file_name = f"{photo_id}.{file_extension}"
    file_path = os.path.join(save_dir, file_name)

    # Пропускаем, если уже скачано
    if os.path.exists(file_path):
        continue

    # Загружаем изображение
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            # Записываем последний загруженный photo_id в progress_file
            with open(progress_file, "w") as f:
                f.write(photo_id)

            print(f"Скачано: {photo_id}")

    except Exception as e:
        print(f"Ошибка при загрузке {photo_id}: {e}")

print(f"Все изображения сохранены в папку: {save_dir}")
