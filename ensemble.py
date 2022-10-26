from sklearn.tree import DecisionTreeRegressor, plot_tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_excel('hh_dataset.xlsx')

features = ['salary', 'per_hour']

df = df[features + ['target_value']]
X = df[features]
y = df["target_value"]

df1 = df.sample(n=500, replace=True)
X1 = df1[features]
y1 = df1["target_value"]
df2 = df.sample(n=500, replace=True)
X2 = df2[features]
y2 = df2["target_value"]
df3 = df.sample(n=500, replace=True)
X3 = df3[features]
y3 = df3["target_value"]
df4 = df.sample(n=500, replace=True)
X4 = df4[features]
y4 = df4["target_value"]
df5 = df.sample(n=500, replace=True)
X5 = df5[features]
y5 = df5["target_value"]


tree1 = DecisionTreeRegressor(random_state=1)
tree2 = DecisionTreeRegressor(random_state=1)
tree3 = DecisionTreeRegressor(random_state=1)
tree4 = DecisionTreeRegressor(random_state=1)
tree5 = DecisionTreeRegressor(random_state=1)

tree1.fit(X1, y1)
tree2.fit(X2, y2)
tree3.fit(X3, y3)
tree4.fit(X4, y4)
tree5.fit(X5, y5)

pdf = pd.DataFrame([[1980, 16]], columns=features)
print("")
print(f"tree1 prendict: {tree1.predict(pdf)}\ntree2 prendict: {tree2.predict(pdf)}\ntree3 prendict: {tree3.predict(pdf)}\ntree4 prendict: {tree4.predict(pdf)}\ntree5 prendict: {tree5.predict(pdf)}\n")
print(f"Average predict: {(tree1.predict(pdf) + tree2.predict(pdf) + tree3.predict(pdf) + tree4.predict(pdf) + tree5.predict(pdf))/5}")