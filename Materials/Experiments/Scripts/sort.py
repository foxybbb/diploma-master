import os
import pandas as pd
import re

# Корень поиска
root_dir = "."

# Категории и соответствующие DataFrame'ы
groups = {
    "coridor": [],
    "coridorangle": [],
    "men1": [],
    "men2": []
}

# Поиск и распределение файлов
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith("_last_data_point.csv"):
            file_path = os.path.join(dirpath, filename)

            # Определение категории
            for group in groups:
                if f"/{group}/" in file_path.replace("\\", "/"):
                    try:
                        df = pd.read_csv(file_path)

                        # Извлечение расстояния из имени файла
                        match = re.search(r'(\d+)', filename)
                        if match:
                            distance = int(match.group(1))
                            df['distance'] = distance
                            df['source_file'] = file_path  # можно убрать
                            groups[group].append(df)
                    except Exception as e:
                        print(f"❌ Ошибка при обработке файла {file_path}: {e}")
                    break  # не проверять другие группы

# Сохранение файлов
for group, data_list in groups.items():
    if data_list:
        merged_df = pd.concat(data_list, ignore_index=True)
        filename = f"{group}_data.csv"
        merged_df.to_csv(filename, index=False)
        print(f"✅ Сохранено: {filename}")
    else:
        print(f"⚠️ Нет данных для группы: {group}")
