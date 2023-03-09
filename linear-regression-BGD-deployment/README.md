# Linear Regression Model

## 1.  Batch Gradient Descent algorithm
### a. Model: 
$$ f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b $$

### b. Cost function:
$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$ 

### c. Gradient descent: 
Repeat until convergence: 

$$\begin{align*}
w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \\ 
b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}
\end{align*}$$

where: 
* n is the number of features
* $w_j$ and $b$, are updated simultaneously and

$$\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}  +  \frac{\lambda}{m} w_j\\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
\end{align*}$$
