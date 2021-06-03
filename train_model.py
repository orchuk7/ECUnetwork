import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
import logging
import pandas as pd


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

df = pd.read_csv('Data.csv', delimiter = ";")

print(df.head())

df = df.sample(frac=1).reset_index(drop=True)

print(df.head())
X = df.iloc[:, : -1].values
y = df.iloc[:, -1].values

print(X.shape, end="\n")
print(y.shape)

X_train = X[:6000, :]
X_test = X[6000:, :]

y_train = y[:6000]
y_test = y[6000:]

print(y_test)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train -= mean
X_train /= std
X_test -= mean
X_test /= std

nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Dense(8, activation='relu', input_shape=([8])))
nn.add(tf.keras.layers.Dense(1024, activation='relu'))
nn.add(tf.keras.layers.Dropout(0.5))
nn.add(tf.keras.layers.Dense(1024, activation='relu'))
nn.add(tf.keras.layers.Dropout(0.5))
nn.add(tf.keras.layers.Dense(2048, activation='relu'))
nn.add(tf.keras.layers.Dense(1, activation='relu'))

nn.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

nn.summary()

history = nn.fit(X_train, y_train, batch_size = 80, epochs = 150, validation_split=0.2)

nn.save('model')

plt.plot(history.history['mae'], 
         label='Середня абсолютна похибка на навчальному наборі')
plt.plot(history.history['val_mae'], 
         label='Середня абсолютна похибка на перевірочному наборі')
plt.xlabel('Епоха навчання')
plt.ylabel('Средня абсолютна похибка')
plt.legend()
plt.show()
plt.savefig('./graphs.png')

predictions = nn.predict(X_test)
print(predictions)
for i in range(0,len(y_test)):
	print("Точне значення: {0}, передбачуване значення: {1}".format(y_test[i], predictions[i][0]))
