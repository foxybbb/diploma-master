import pandas as pd
import matplotlib.pyplot as plt

# Загрузка men1 и men2 данных
men1_df = pd.read_csv("men1_data.csv")
men2_df = pd.read_csv("men2_data.csv")

# Сортировка по расстоянию
men1_df = men1_df.sort_values(by="distance")
men2_df = men2_df.sort_values(by="distance")

# Построение графика: Radar Height (peak_value) vs Distance
plt.figure(figsize=(10, 6))
plt.plot(men1_df['distance'], men1_df['peak_value'], marker='o', label='1 meter')
plt.plot(men2_df['distance'], men2_df['peak_value'], marker='s', label='2 meter')

plt.xlabel('Radar Height (mm)')
plt.ylabel('Radar Signal Peak Value')
plt.title('Radar Height vs Distance to person')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
