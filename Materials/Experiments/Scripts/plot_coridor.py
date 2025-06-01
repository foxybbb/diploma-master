import pandas as pd
import matplotlib.pyplot as plt

# Загрузка men1 и men2 данных
men1_df = pd.read_csv("coridorangle_data.csv")

# Сортировка по расстоянию
men1_df = men1_df.sort_values(by="distance")

# Построение графика: Radar Height (peak_value) vs Distance
plt.figure(figsize=(10, 6))
plt.plot(men1_df['distance'], men1_df['peak_value'], marker='o', label='Peak Value')
plt.plot(men1_df['distance'], men1_df['average_value'], marker='o', label='Average value')
plt.plot(men1_df['distance'], men1_df['dist'], marker='o', label='Distance')

plt.xlabel('Angle (degrees)')
plt.ylabel('Radar Signal Value')
plt.title('Radar Signal vs Angle')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
