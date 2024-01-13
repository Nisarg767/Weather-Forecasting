import pandas as pd
import numpy as np
import math
from pyspark.sql import SparkSession

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# import sklearn.metrics as metrics


spark = SparkSession.builder.appName("Weather Prediction").config("spark.executor.memory", "6gb").getOrCreate()

file_path = "./Daily_Temperatures_Jena.csv"

df = spark.read.format("csv").option("header", "true").option("sep", ",").load(file_path)

avg_temp = np.array(df.select('Tmean').collect())
avg_temp = np.asarray(avg_temp, dtype=float).reshape(-1)


training_size=int(len(avg_temp)*0.70)
test_size=len(avg_temp)-training_size
train_data, test_data = avg_temp[0:training_size], avg_temp[training_size: len(avg_temp)]



time_step = 60

features = 1

def create_dataset(dataset):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step)] 
		dataX.append(a)
		dataY.append(dataset[i + time_step])
	return np.array(dataX), np.array(dataY)

x_train, y_train = create_dataset(train_data)


def reshape_data(x):
    return x.reshape((x.shape[0], x.shape[1], features))

x_train = reshape_data(x_train)
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(time_step, features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=32)


x_test, y_test = create_dataset(test_data)

x_test = reshape_data(x_test)
y_hat = model.predict(x_test)

MSE = np.square(np.subtract(y_test, y_hat)).mean() 


# rmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat))
print(f"RMSE: {math.sqrt(MSE)}")
