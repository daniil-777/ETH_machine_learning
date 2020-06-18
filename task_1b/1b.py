import pandas as pd
import numpy as np
from numpy.core.multiarray import dtype
from sklearn.linear_model import BayesianRidge

# Library versions:
#     pandas 1.0.4
#     numpy 1.18.5
#     sklearn 0.23.1

def get_data(file_name):
    train = pd.read_csv(file_name)
    X = train.iloc[:, 2:7].to_numpy()
    Y = train.iloc[:, 1].to_numpy()

    return Y, X

def transform_features(X):
    linear = X
    quad = np.square(X)
    exp = np.exp(X)
    cos = np.cos(X)
    const = np.ones(shape=(X.shape[0], 1), dtype=np.float64)
    return np.concatenate((linear, quad, exp, cos, const), axis=1)

def bayes_regr(X, Y):
    regr =  BayesianRidge(compute_score=True)
    regr.set_params(alpha_1=10, lambda_1=1e-3)
    regr.fit(X, Y)

    w_hat = regr.coef_
    # Y_pred = regr.predict(X)

    return w_hat

def print_array_to_file(arr, file_name):
    output = pd.DataFrame(arr)
    output.to_csv(file_name, index=False, header=False)


if __name__ == '__main__':
    Y, X = get_data("data/train.csv")

    X = transform_features(X)
    # print_array_to_file(X, "x_transfrom.csv")

    w_hat = bayes_regr(X, Y)
    print_array_to_file(w_hat, "output.csv")
