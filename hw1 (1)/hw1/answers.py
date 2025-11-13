r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
