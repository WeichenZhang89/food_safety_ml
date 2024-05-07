import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/will./Desktop/food_safety_ml/mushrooms.csv')
data.columns = [col.strip() for col in data.columns]
null_data_count = data.isnull().sum().sum()
if (null_data_count == 0):
    print("No missing data")
else:
    print(f"{null_data_count} missing data.")




