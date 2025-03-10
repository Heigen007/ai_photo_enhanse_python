import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
jpeg_webp_results = pd.read_csv("jpeg_webp_results.csv")
ai_results = pd.read_csv("ai_results.csv")

# Объединение данных
combined_results = pd.concat([jpeg_webp_results, ai_results])

# Упорядочиваем изображения
combined_results.sort_values(by=["Image", "Format"], inplace=True)

# Получение уникальных имен изображений
image_names = combined_results["Image"].unique()

# Визуализация графиков
plt.figure(figsize=(10, 5))
for fmt in ["JPEG", "WebP", "AI"]:
    subset = combined_results[combined_results["Format"] == fmt]
    subset = subset.set_index("Image").reindex(image_names).reset_index()
    plt.plot(subset["Image"], subset["PSNR"], label=f"{fmt} PSNR", marker='o')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Image Name")
plt.ylabel("PSNR")
plt.title("PSNR Comparison: JPEG vs WebP vs AI")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for fmt in ["JPEG", "WebP", "AI"]:
    subset = combined_results[combined_results["Format"] == fmt]
    subset = subset.set_index("Image").reindex(image_names).reset_index()
    plt.plot(subset["Image"], subset["SSIM"], label=f"{fmt} SSIM", marker='o')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Image Name")
plt.ylabel("SSIM")
plt.title("SSIM Comparison: JPEG vs WebP vs AI")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for fmt in ["JPEG", "WebP", "AI"]:
    subset = combined_results[combined_results["Format"] == fmt]
    subset = subset.set_index("Image").reindex(image_names).reset_index()
    plt.plot(subset["Image"], subset["Compressed Size (KB)"], label=f"{fmt} Size", marker='o')
plt.xticks(rotation=45, ha='right')
plt.xlabel("Image Name")
plt.ylabel("Size (KB)")
plt.title("File Size Comparison: JPEG vs WebP vs AI")
plt.legend()
plt.show()