r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
### **1) False**  
Splitting the data with a bad ratio may not allow for enough training examples, unbalanced distributions can harm generalization, and issues like temporal leakage (e.g., training on future data and testing on past data) can introduce overfitting. In general, we want the split to preserve the properties of the full dataset, such as distribution, noise, temporal dependencies, etc.

---

### **2) False**  
The test set should not be used during cross-validation because tuning parameters based on test performance causes overfitting to the test set. This prevents the test set from serving as an unbiased measure of true generalization.

---

### **3) True**  
Cross-validation uses validation performance on each fold as a proxy for generalization error on unseen data. The test set is then used only once at the end to obtain a final, unbiased estimate of performance on truly unseen data.

---

### **4) True**  
Injecting noise into labels helps reveal robustness because a strong model should maintain reasonable validation performance despite small amounts of corrupted data. A non-robust model will overfit the noisy labels and perform poorly on the validation set, exposing sensitivity and lack of generalization.



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**No. This is a terrible idea! Using the test set to pick the best parameters would lead to overfitting as the test-error is no longer "error on unseen data".**

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**



If we allow $\Delta < 0$, the loss function loses its intended geometric
meaning as a *margin*-based separation criterion.

From the given definition,
$$
L(\mat{W}) =
\frac{1}{N} \sum_{i=1}^{N} L_i(\mat{W})
+ \frac{\lambda}{2}|\mat{W}|^2,
\quad
L_i(\mat{W}) = \sum_{j \neq y_i} \max\!\left(0,\; \Delta + \vectr{w_j}\vec{x_i}
- \vectr{w_{y_i}}\vec{x_i}\right),
$$
a **positive** $\Delta$ enforces that the correct class score
$\vectr{w_{y_i}}\vec{x_i}$ must exceed all incorrect class scores
$\vectr{w_j}\vec{x_i}$ by at least a fixed margin $\Delta > 0$.
If this condition is violated, the hinge term becomes positive and penalizes the model, 
driving it to enlarge the difference between the correct and incorrect class scores.

However, if $\Delta < 0$, the margin constraint is reversed:
$$
\vectr{w_{y_i}}\vec{x_i} \ge \vectr{w_j}\vec{x_i} - |\Delta|,
$$
which can easily hold even when $\vectr{w_{y_i}}\vec{x_i} < \vectr{w_j}\vec{x_i}$.
Thus, the hinge terms may all become zero even for misclassified samples,
causing $L(\mat{W})$ to vanish although the model performs poorly.

In this case, the classifier receives no gradient signal to correct its mistakes,
and the optimization degenerates into minimizing only the regularization term
$\tfrac{\lambda}{2}|\mat{W}|^2$, which simply shrinks the weights toward zero
without improving class separation.

Hence, a negative $\Delta$ destroys the fundamental purpose of the SVM.

"""

part2_q2 = r"""
**Your answer:**

The linear model learns template-like weight images for each digit, 
highlighting the average shape of that class. 
Bright areas correspond to pixels that increase the score for a given class, 
while dark areas reduce it. Because the model is linear, 
it can only separate digits based on overall intensity patterns rather than specific features or stroke shapes.

The observed errors occur because these digits share similar pixel distributions or shapes. 
ns, it confuses such visually overlapping digits. For example, the model sometimes predicts 6 when the actual digit is 5, 
because the two have very similar shapes. The main difference is that a typical 6 has a closed lower loop, while a 5’s lower curve is open. 
In this case, the handwritten 5 had a nearly closed bottom loop, so the model confused it for a 6.


"""

part2_q3 = r"""
**Your answer:**

(1) The learning rate is good. The training loss decreases quickly and steadily.
    If the learning rate were too low, the loss would drop very slowly (the curve would look almost flat). 
    If it were too high, the loss would jump up and down wildly or even increase sometimes.



(2) The training and validation accuracy curves rise together and stay close throughout training, without a large gap between them. 
This means the model generalizes well to unseen data, so it is not overfitted.

However, since the validation accuracy remains slightly below the training accuracy, 
it indicates that the model still performs a bit better on the data it has already seen. 
This small gap means the model hasn’t fully captured all the underlying structure in the data.

This indicates slight underfitting: the model could potentially perform better with a more complex architecture or longer training, but it already generalizes well and is not overfitting.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Ideally we would want the residuals to be as close to zero as possible**, meaning that we managed to predict the target well. This can be expressed mathematically as:

$$y - \hat{y} = 0 \quad \Rightarrow \quad y = \hat{y}$$

That means we want to see a **very thin cloud of points around** $y=0$, with the red lines (representing the standard deviation of the residuals) being relatively close to zero.

### Top 5 Features Model

As we can see in the plot for the top 5 features, the residuals are concentrated between $-10$ and $10$. However, there are a lot of examples that are further away from $0$ (up to $\pm 20$), indicating that:
- The model is not able to predict the target perfectly
- The model is still able to predict the target somewhat well, as the residuals are concentrated around $0$

### After Cross-Validation (CV)

In the final plot after CV, the residuals are:
- **Closer to** $0$ and more concentrated (lower standard deviation)
- The red lines are closer to $0$, indicating that the standard deviation of the residuals is smaller

This indicates that the model is able to predict the target better after CV and feature engineering.

### Generalization

It is also really nice to see that on the test set we get similar results (although understandably slightly worse) to the training set, indicating that **the model generalizes well**.

### Conclusion

All that is to say that:
- **CV + feature engineering** seemed to improve the model's performance
- A **linear regression** fits the data relatively well
- The model performs **even better** when introducing non-linear features
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**1. Is this still a linear regression model? Why or why not?**

No, it is not a linear regression model anymore (at least not as defined in this notebook). In the notebook we defined a linear regression model as:

$$\hat{y} = \vec{w}^T\vec{x} + b$$

where the prediction $\hat{y}$ is a **linear function** of the original features $\vec{x}$.

When we add non-linear features (e.g., $\vec{\tilde{x}} = (x_1, x_2, x_1^2, x_1 x_2, x_2^2)$), the model becomes:

$$\hat{y} = \vec{w}^T\vec{\tilde{x}} + b$$

This is still **linear in the transformed features** $\vec{\tilde{x}}$, but it is **non-linear in the original features** $\vec{x}$. For example, if $\tilde{x}_3 = x_1^2$, then the prediction includes terms like $w_3 x_1^2$, which is quadratic in the original feature $x_1$.

That means that the model is no longer assuming a linear relationship between the **original** features and the target, which is the definition of a linear regression model.

**2. Can we fit any non-linear function of the original features with this approach?**

**No, we cannot fit any non-linear function** (if by "this approach" you mean the approach of adding a finite number of non-linear features to the model).

We can only model functions that can be expressed as a linear combination of the features we engineer. For example, if we only create polynomial features up to degree $d$, we can only fit polynomial functions of degree at most $d$.

In a course I took last year (Deep Learning and Approximation Theory) we showed that this space is not dense in the continuous functions space. (The proof is not hard, but it is not the point of this course.)

**3. How would adding non-linear features affect the decision boundary of a linear classifier?**

In a linear classification model, the decision boundary is defined by a hyperplane. Introducing non-linear features would still yield a hyperplane in the transformed feature space, which will translate to a **non-linear decision boundary** in the original feature space.


"""

part3_q3 = r"""
**1.**

$$\mathbb{E}_{x,y}[|y-x|] = \int_0^1 \int_0^1 |y-x| \, dy \, dx$$

$$\int_0^1 |y-x| \, dy = \int_0^x (x-y) \, dy + \int_x^1 (y-x) \, dy$$

$$\int_0^x (x-y) \, dy = \left[xy - \frac{y^2}{2}\right]_0^x = x^2 - \frac{x^2}{2} = \frac{x^2}{2}$$

$$\int_x^1 (y-x) \, dy = \left[\frac{y^2}{2} - xy\right]_x^1 = \left(\frac{1}{2} - x\right) - \left(\frac{x^2}{2} - x^2\right) = \frac{1}{2} - x + \frac{x^2}{2}$$

$$\int_0^1 |y-x| \, dy = \frac{x^2}{2} + \frac{1}{2} - x + \frac{x^2}{2} = x^2 - x + \frac{1}{2}$$

$$\mathbb{E}_{x,y}[|y-x|] = \int_0^1 \left(x^2 - x + \frac{1}{2}\right) dx = \left[\frac{x^3}{3} - \frac{x^2}{2} + \frac{x}{2}\right]_0^1 = \frac{1}{3} - \frac{1}{2} + \frac{1}{2} = \frac{1}{3}$$

**Answer: $\mathbb{E}_{x,y}[|y-x|] = \frac{1}{3}$**

---

**2. What is the expected value $\mathbb{E}_{x}[|\hat{x}-x|]$? Answer should be a polynomial of $\hat{x}$.**

Very similar to the previous calculation, but now we treat $\hat{x}$ as a fixed value.

$$\mathbb{E}_{x}[|\hat{x}-x|] = \int_0^1 |\hat{x}-x| \, dx$$

$$\mathbb{E}_{x}[|\hat{x}-x|] = \int_0^{\hat{x}} (\hat{x}-x) \, dx + \int_{\hat{x}}^1 (x-\hat{x}) \, dx$$

$$\int_0^{\hat{x}} (\hat{x}-x) \, dx = \left[\hat{x}x - \frac{x^2}{2}\right]_0^{\hat{x}} = \hat{x}^2 - \frac{\hat{x}^2}{2} = \frac{\hat{x}^2}{2}$$

$$\int_{\hat{x}}^1 (x-\hat{x}) \, dx = \left[\frac{x^2}{2} - \hat{x}x\right]_{\hat{x}}^1 = \left(\frac{1}{2} - \hat{x}\right) - \left(\frac{\hat{x}^2}{2} - \hat{x}^2\right) = \frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2}$$

Therefore:

$$\mathbb{E}_{x}[|\hat{x}-x|] = \frac{\hat{x}^2}{2} + \frac{1}{2} - \hat{x} + \frac{\hat{x}^2}{2} = \hat{x}^2 - \hat{x} + \frac{1}{2}$$

And if $\hat{x} \ is an estimator s.t it is between 0 and 1 this answer stands. However, if $\hat{x}$ is not between 0 and 1, we need to consider the other cases.

**Case 2: $\hat{x} < 0$**

For all $x \in [0,1]$, we have $|\hat{x}-x| = x-\hat{x}$:

$$\mathbb{E}_{x}[|\hat{x}-x|] = \int_0^1 (x-\hat{x}) \, dx = \left[\frac{x^2}{2} - \hat{x}x\right]_0^1 = \frac{1}{2} - \hat{x}$$

**Case 3: $\hat{x} > 1$**

For all $x \in [0,1]$, we have $|\hat{x}-x| = \hat{x}-x$:

$$\mathbb{E}_{x}[|\hat{x}-x|] = \int_0^1 (\hat{x}-x) \, dx = \left[\hat{x}x - \frac{x^2}{2}\right]_0^1 = \hat{x} - \frac{1}{2}$$

---

**3. Explain why we can drop the value of the scalar of the polynomial?**
**Constants don't affect the location of the minimum**, since the derivative of a constant is zero, so adding or subtracting a constant doesn't change where the derivative equals zero (the critical points).

"""

# ==============

# ==============
