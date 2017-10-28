import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

training_set = pd.read_csv('data/Google_Stock_Price_Train.csv')
series = training_set.iloc[:,1]

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(series, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)
