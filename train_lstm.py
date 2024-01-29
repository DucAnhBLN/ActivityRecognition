import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

# Đọc dữ liệu

vaytay_df = pd.read_csv("vaytay.txt")# 2
lacnguoi_df = pd.read_csv("lacnguoi.txt") # 3
votay_df = pd.read_csv("votay.txt") #4


X = []
y = []
no_of_timesteps = 10




dataset = votay_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)


dataset = lacnguoi_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)


dataset = vaytay_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(2)



X, y = np.array(X), np.array(y)
encoder = LabelBinarizer()
y = encoder.fit_transform(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 3, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
model.save("model.h5")


