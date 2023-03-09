# Linear Regression Model

## Model deployment with Sklearn package 
The steps that we perform throughout model training and evaluation:
### a. Data preparation
- Split model into train, test, split
- Perform feature scaling
### b. Model buidling
- SGD Regresssor
- Linear Regression

Perform polynominal feature selection for both models, and compare MSE of cross validation set to choose the best model

### c. Model selection

- On comparing MSE of LinearRegression and SGDRegressor on cross validation set, LinearRegression gives a smaller MSE

- The best model is Linear Regression, polynomial feature of degree 3.

### d. Diagnose performance using learning curve

- The following learning curve calculated using the training dataset to give an idea of how well the model is learning

![Alt text](img/learning-curve.png)

### Results: 
- The model has both high bias and high variance problem
- The gap for the training and validation curve becomes mall as the training dataset size increases. This indicates that adding more examples to our model is not going to improve its performance. 
- The training MSE is very high, which indicates a high variance problem

### To fix these problems,
- Add more features
- Train different models