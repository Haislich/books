# Computing Translation

Given matched points \( (x_i, y_i) \) in \( A \) and \( (x'_i, y'_i) \) in \( B \), the translation equations are:

$$
x'_i = x_i + t_x \\
y'_i = y_i + t_y
$$

Rearranging gives:

$$
t_x = x'_i - x_i \\
t_y = y'_i - y_i
$$

To compute \( t_x \) and \( t_y \) across all matches, the least squares approach minimizes the sum of squared differences:

$$
\min_{t_x, t_y} \sum_{i=1}^n \left((x'_i - x_i - t_x)^2 + (y'_i - y_i - t_y)^2\right)
$$

The solution for \( t_x \) and \( t_y \) is:

$$
\hat{t}_x = \frac{1}{n} \sum_{i=1}^n (x'_i - x_i), \quad \hat{t}_y = \frac{1}{n} \sum_{i=1}^n (y'_i - y_i)
$$

This averages the translation estimates from all matches, reducing the impact of noise.
