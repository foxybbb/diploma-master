import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
indoor_df = pd.read_csv("indoor_data_combined.csv")
outdoor_df = pd.read_csv("outdoor_data_combined.csv")

# Сортировка по расстоянию
indoor_df = indoor_df.sort_values(by="distance")
outdoor_df = outdoor_df.sort_values(by="distance")

# Построение графика
plt.figure(figsize=(10, 6))

# Indoor
plt.plot(indoor_df['distance'], indoor_df['peak_value'], marker='o', label='Indoor - Peak Value')
plt.plot(indoor_df['distance'], indoor_df['average_value'], marker='s', label='Indoor - Average Value')

# Outdoor
plt.plot(outdoor_df['distance'], outdoor_df['peak_value'], marker='^', linestyle='--', label='Outdoor - Peak Value')
plt.plot(outdoor_df['distance'], outdoor_df['average_value'], marker='d', linestyle='--', label='Outdoor - Average Value')

# Оформление
plt.xlabel('Distance (mm)')
plt.ylabel('Signal Magnitude')
plt.title('Peak and Average Signal Magnitude vs Distance\nIndoor vs Outdoor')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
