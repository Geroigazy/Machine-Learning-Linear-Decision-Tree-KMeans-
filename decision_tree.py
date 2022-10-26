from sklearn.tree import DecisionTreeRegressor, plot_tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mse(true, pred):
    return np.mean(np.square(true - pred))


df = pd.read_excel('hh_dataset.xlsx')

features = ['salary', 'per_hour']

df = df[features + ['target_value']]
X = df[features]
y = df["target_value"]


tree = DecisionTreeRegressor(random_state=1)
tree.fit(X, y)
pdf = pd.DataFrame([[2000, 16]], columns=features)
plt.figure(figsize=(10, 15), dpi=720)
plot_tree(tree, feature_names=features, filled=True, max_depth=5)
plt.savefig("fig.png")
print(tree.predict(pdf))