import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

training_set = pd.read_csv('data/Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

x_train = training_set[:-1]
y_train = training_set[1:]

num_obs = len(x_train)
timestep = 1
num_input_features = 1
x_train = np.reshape(x_train,(num_obs,timestep,num_input_features))

#Sequential class helps us to initialize our model as sequence of layers
#in contrast to boltzmann machines where we create a graph
from keras.models import Sequential
#Dense class which helps to create output layer of RNN
from keras.layers import Dense
#To build lstm layers of the network
from keras.layers import LSTM

regressor = Sequential()
#Add function add layers
#Units tell how many memory units in LSTM Network
regressor.add(LSTM(units = 4, activation = 'sigmoid' , input_shape = (timestep,num_input_features)))
#units here tell number of neurons in output layer
regressor.add(Dense(units=1)) 
#Compiling the RNN
regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error')

regressor.fit(x_train,y_train, batch_size = 32, epochs = 200)


test_set = pd.read_csv('data/Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

'''
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(len(inputs),timestep,num_input_features))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


real_stock_price = real_stock_price[1:]
predicted_stock_price = predicted_stock_price[:-1]
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))
'''

outp = real_stock_price[0]
answers = []
for i in range(0,19):
	inputval = outp
	temp = sc.transform(inputval.reshape(1,-1))
	temp = np.reshape(temp,(len(temp),timestep,num_input_features))
	predicted_stock_price = regressor.predict(temp)
	outp = sc.inverse_transform(predicted_stock_price)
	answers.append(outp[0][0])

predicted_stock_price = np.array(answers,dtype='float32')
real_stock_price = real_stock_price[1:]
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()





