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
import plotly.graph_objects as go

import pandas as pd
df = pd.read_csv('data_permuted.csv')

fig = go.Figure([go.Scatter(x=df['Real Power (P)'], y=df['Apparent Power (S)'])])
fig.show()
