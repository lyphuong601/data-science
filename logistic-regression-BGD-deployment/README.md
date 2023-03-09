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
