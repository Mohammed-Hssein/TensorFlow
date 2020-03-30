"""
Created on Mon Mar 30 19:50:07 2020

@author: mohammed

Linear Regression using TensorFLow ....

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loding the dataset
dataset = pd.read_csv('FuelConsumption.csv')
x = dataset.iloc[:, 4].values
y = dataset.iloc[:, [12]].values

#Scale y axis :
scaler = StandardScaler()
y = scaler.fit_transform(y)
y = y.reshape(-1)

#Trainning set and test set
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 1/5, random_state = 0)
plt.scatter(trainX, trainY)

#Trying to find best a and b such that Y = a X + b using tendorFLow

#Start with random values
a = tf.Variable(2.0)
b = tf.Variable(-2.5)
y = a * trainX + b

#Define the loss to minimoze 
loss = tf.reduce_mean(tf.square(y - trainY))

#Define optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)

#Define operation 
trainOperation = optimizer.minimize(loss)

#Start  TensorFLow session to apply LInear Regression model

# iniitalize variables
init = tf.global_variables_initializer()
#Start Session
session = tf.Session()
session.run(init)

#COmpute LInear REgression
lossValues = []
abValeus = []
epochsTraining = 20000


for step in range(epochsTraining):
    _,lossValue,aValue,bValue = session.run([trainOperation, loss, a, b])
    lossValues.append(lossValue)
    abValeus.append([aValue, bValue])
    
    if step%1000 == 0 :
        print("loss is :" + str(lossValue), "    a value is :" + str(aValue), 
              "    b value is :"+str(bValue))



# NOW a and b iare readdy to make predictions