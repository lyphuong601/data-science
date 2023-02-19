import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
np.set_printoptions(precision=2)


def plot_train_cv_mses(degree, train_mse, cv_mse, baseline):
    plt.plot(degree, train_mse, marker='o', c='r', label='training MSEs')
    plt.plot(degree, cv_mse, marker='o', c='b', label='CV MSEs')
    plt.plot(degree, np.repeat(baseline, len(degree)),
             linestyle='--', label='baseline')
    plt.title("degree of polynomial vs. train and CV MSEs")
    plt.xticks(degree)
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def train_poly(model_t, x_train, y_train, x_cv, y_cv, max_degree=10):

    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    degrees = range(1, max_degree+1)

    for degree in degrees:
        model = copy.deepcopy(model_t)
        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        scaler_poly = StandardScaler()  # Scale the training set
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        model.fit(X_train_mapped_scaled, y_train)  # Create and train the model
        models.append(model)
        # Compute training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features and scale the cross-validation set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
        # Compute cross-validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

    return degrees, train_mses, cv_mses, scalers, models
