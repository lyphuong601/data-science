# Logistic Regression Model

## 1. Batch Gradient Descent algorithm
### a. Model: 
  $$f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b)$$

  where $g(z)$ is known as the sigmoid function and it maps all input values to values between 0 and 1:

  $$g(z) = \frac{1}{1+e^{-z}}$$

### b. Cost function

Combine the losses to form the cost of all examples

$$ J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right]  + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$

where:
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is:

    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
    
*  m is the number of training examples in the data set and:

### c. Gradient descent: 
Repeat until convergence: 

$$\begin{align*}
w_j &= w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j}  \\ 
b &= b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}
\end{align*}$$

where: 
* n is the number of features
* $w_j$ and $b$, are updated simultaneously and

$$\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}  +  \frac{\lambda}{m} w_j\\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
\end{align*}$$

## 2. Model deployment with Sklearn package 
The steps that we perform throughout model training and evaluation:
### a. Data preparation
- Split model into train, test, split
- Perform feature scaling
### b. Model buidling
- SGD Classifier
- Logistic Regression

=> Perform polynominal feature selection for both models, and compare MSE of cross validation set to choose the best model

### c. Model selection
- Use ROC curve to demonstrate:
    - The tradeoff between sensitivity and specificity
    - The closer the ROC curve follows the left-hand border and the top border, the more accurate the model
    - The larger the area under ROC, the better the model

![Alt text](img/roc.png)

### Results: 
- On comparing LogisticRegression and SGDClassifier for this dataset, LogisticRegression gives a bigger ROC area under the curve. So, we choose LogisticRegression

- The best model is Logistic Regression, polynomial feature of degree 1 and the reported MSE is 0.09 (a 2.5% degree in MSE)

### d. Diagnose performance using learning curve
- The following learning curve calculated using the training dataset to give an idea of how well the model is learning

![Alt text](img/learning-curve.png)

### Results: 
- The model has high bias problem
- The gap for the training and validation curve becomes extremely small as the training dataset size increases. This indicates that adding more examples to our model is not going to improve its performance. 

### Too fix high bias problem:
- Add more features
- Decrease regularization 