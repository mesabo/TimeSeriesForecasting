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

import pandas as pd

# Load the text file into a DataFrame
df = pd.read_csv('data_permuted.txt', delimiter=',')  # Specify the delimiter if it's not comma-separated

# Save the DataFrame to a CSV file
df.to_csv('data_permuted.csv', index=False)  # Specify index=False to exclude row numbers in the CSV file



#
# # Load the CSV file into a DataFrame
# df = pd.read_csv('data_warped.csv')  # Adjust the delimiter if necessary
#
# # Column titles
# column_titles = ['Real Power (P)', 'Real Energy (Pt)', 'Reactive Power (Q)',
#                  'Reactive Energy (Qt)', 'Apparent Power (S)', 'Apparent Energy (St)',
#                  'day_of_week', 'month']
#
# # Rename the columns
# df.columns = column_titles
#
# # Save DataFrame to CSV with the new column names
# df.to_csv('data_warped.csv', index=False)
