import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_curve, auc, roc_auc_score, precision_recall_curve, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def check_missing_data(data: pd.DataFrame) -> pd.Series:
    """Check for missing data in the df (display in descending order)"""
    result = ((data.isnull().sum() * 100) /
              len(data)).sort_values(ascending=False)
    return result


def plot_train_cv_mses(degrees, train_mses, cv_mses, baseline=None):
    plt.plot(degrees, train_mses, marker='o', c='r', label='training MSEs')
    plt.plot(degrees, cv_mses, marker='o', c='b', label='CV MSEs')
    plt.plot(degrees, np.repeat(baseline, len(degrees)),
             linestyle='--', label='baseline')
    plt.title("degree of polynomial vs. train and CV MSEs")
    plt.xticks(degrees)
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


def train_poly(model_t, x_train, y_train, x_cv, y_cv, max_degree=5):

    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    degrees = range(1, max_degree+1)

    # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
    for degree in degrees:
        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Create and train the model
        model = copy.deepcopy(model_t)
        model.fit(X_train_mapped_scaled, y_train)
        models.append(model)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features and scale the cross-validation set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute the cross-validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

    return degrees, train_mses, cv_mses, scalers, models


def plot_roc(models, x_test, y_test):
    for m in models:
        model = m['model']  # select the model
        # model.fit(x_train, y_train)  # train the model
        y_pred = model.predict(x_test)  # predict the test data
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(
            x_test)[:, 1])  # Compute False postive rate, and True positive rate
        # Calculate Area under the curve to display on the plot
        auc = roc_auc_score(y_test, model.predict(x_test))

        # Now, plot the computed values
        plt.plot(fpr, tpr, label=f'{m["label"]} (roc auc: {auc:0.2f})')
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.text(0.6, 0.5, "Baseline")
    plt.text(0.3, 0.8, "ROC Curve")
    plt.legend(loc="lower right")
    plt.show()   # Display


def plot_auc(models, x_test, y_test):
    for m in models:
        model = m['model']  # select the model
        # model.fit(X_train_scaled, y_train)  # train the model
        y_pred = model.predict(x_test)  # predict the test data
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, model.predict_proba(
            x_test)[:, 1])  # Compute False postive rate, and True positive rate
        # Calculate Area under the curve to display on the plot
        lr_f1, lr_auc = f1_score(y_test, y_pred), auc(lr_recall, lr_precision)
        # Now, plot the computed values
        plt.plot(lr_recall, lr_precision,
                 label=f'{m["label"]} (auc: {lr_auc:0.2f}, f1: {lr_f1:0.2f})')

    # Custom settings for the plot
    base = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [base, base], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC')
    plt.text(0.5, 0.3, "Baseline")
    plt.text(0.3, 0.8, "AUC Curve")
    plt.legend(loc="lower right")
    plt.show()   # Display


def evaluate_model(y_train, yhat, y_test, yhat_test):
    print(f"Training MSE: {(mean_squared_error(y_train, yhat) / 2):.5f}")
    print(f"Test MSE: {(mean_squared_error(y_test, yhat_test) / 2):.5f}")
    print(f"Test set F1 score: {f1_score(y_test, yhat_test):.5f}")
    print(f"Test set accuracy score: {recall_score(y_test, yhat_test):.5f}")