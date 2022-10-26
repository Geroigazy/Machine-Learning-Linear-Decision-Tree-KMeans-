import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_excel('hh_dataset.xlsx')
# Multiple linear
# X = df.drop(columns=['target_value', 'target_class'])
# y = df.drop(columns=['hours', 'per_hour', 'salary', 'degree', 'position', 'experience', 'target_class'])
# model = LinearRegression()
# model.fit(X, y)

# print(model.predict(X[:1]))



X = df.drop(columns=['hours', 'salary', 'degree', 'position', 'experience', 'target_class', 'target_value'])
y = df.drop(columns=['hours', 'per_hour', 'degree', 'position', 'experience', 'target_class', 'salary'])

model = LinearRegression()
model.fit(X, y)


model_a = model.coef_[0]
model_b = model.intercept_
print(model.predict(X[:1]))
fig = plt.figure(figsize=(10, 6))

x = np.arange(4, 21)
model_y_sk = model_a * x + model_b

plt.plot(x, model_y_sk, linewidth=2, c='r')
plt.scatter(X, y) 
plt.grid()
plt.xlabel('feature')
plt.ylabel('target')
plt.legend(prop={'size': 16})
plt.show()