import pandas as pd
import glob
import re

# Список всех подходящих CSV-файлов в директории
file_paths = glob.glob("*_last_data_point.csv")

merged_data = []

for path in file_paths:
    # Извлекаем значение расстояния из имени файла
    match = re.search(r'(\d+)mm', path)
    if match:
        distance = int(match.group(1))
        df = pd.read_csv(path)
        df['distance'] = distance  # добавляем колонку distance
        merged_data.append(df)

# Объединяем все DataFrame в один
merged_df = pd.concat(merged_data, ignore_index=True)

# Сохраняем в CSV
merged_df.to_csv("merged_data.csv", index=False)
print("✅ Готово! Объединённый файл: merged_data.csv")
