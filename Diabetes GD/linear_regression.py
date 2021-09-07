import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])

X = diabetes.data
X_train = X[:-30]
X_test = X[-30:]

Y_train = diabetes.target[:-30]
Y_test = diabetes.target[-30:]

algo = linear_model.LinearRegression()
algo.fit(X_train, Y_train)

Y_predict = algo.predict(X_test)

print("mse: ", mean_squared_error(Y_test, Y_predict))

print("Weights: ", algo.coef_)
print("Intercept: ", algo.intercept_)


