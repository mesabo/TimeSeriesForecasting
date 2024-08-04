#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/22/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the CSV file into a DataFrame
# df = pd.read_csv('data_noised.csv')
#
# # Plot the data
# plt.figure(figsize=(8, 6))
# plt.plot(df['Real Power (P)'], df['Reactive Power (Q)'], marker='o', linestyle='-')
# plt.xlabel('Real Power (P)')
# plt.ylabel('Reactive Power (Q)')
# plt.title('Plot of Reactive Power (Q) against Real Power (P)')
# plt.grid(False)
#
# # Save the plot as EPS with higher DPI for better quality
# plt.savefig('data_original.eps', format='eps', dpi=1000)
#
# # Show the plot
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the CSV file into a DataFrame
# df = pd.read_csv('data_noised.csv')
#
# # Plot the data
# plt.figure(figsize=(8, 6))
# plt.plot(df['Real Power (P)'], df['Reactive Power (Q)'], marker='o', linestyle='-', color='orange')  # Change line color to orange
# plt.xlabel('Real Power (P)', color='black')  # Set x-axis label color to black
# plt.ylabel('Reactive Power (Q)', color='black')  # Set y-axis label color to black
# plt.title('Plot of Reactive Power (Q) against Real Power (P)', color='black')  # Set title color to black
# plt.grid(False)
#
# # Save the plot as EPS with higher DPI for better quality
# plt.savefig('data_original.eps', format='eps', dpi=1000)
#
# # Save the plot as PNG with the highest quality possible
# plt.savefig('data_original.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)  # Adjust dpi for desired quality
#
# # Show the plot
# plt.show()


import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('data_noised.csv')

# Select the first 200 rows
df_subset = df

# Plot the data as a bar chart
plt.figure(figsize=(8, 6))
plt.bar(df_subset['Real Power (P)'], df_subset['Apparent Power (S)'], color='orange')
plt.xlabel('Real Power (P)', color='black')  # Set x-axis label color to black
plt.ylabel('Apparent Power (S)', color='black')  # Set y-axis label color to black
plt.title('Bar Chart of Apparent Power (S) against Real Power (P)', color='black')
plt.grid(True)

# Save the plot as EPS with higher DPI for better quality
plt.savefig('./data_aug_tech/data_warped.eps', format='eps', dpi=1000)

# Save the plot as PNG with the highest quality possible
plt.savefig('./data_aug_tech/data_warped.png', format='png', dpi=1000, bbox_inches='tight', pad_inches=0.1)

# Show the plot
plt.show()
