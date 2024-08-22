import numpy as np
import pandas as pd

# import the data required
data = pd.read_csv('Employee data.csv')
print(data.head())


print(data['gender'].unique())
print(data['jobcat'].unique())


data['gender'].value_counts()
data['jobcat'].value_counts()


one_hot_encoded_data = pd.get_dummies(data, columns = ['jobcat', 'gender'])
print(one_hot_encoded_data)
