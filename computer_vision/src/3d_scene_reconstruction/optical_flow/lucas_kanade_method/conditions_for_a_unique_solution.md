# Conditions for a Unique Solution

A unique solution exists if $A^T A$ is invertible. The matrix $A^T A$ is known as the **second moment matrix** (or autocorrelation matrix in some contexts), and it measures the spread or variance of the image gradients within the window. This matrix is symmetric and positive semi-definite.

The eigenvalues of $A^T A$ provide crucial information about the conditioning of the system. In the context of the Lucas-Kanade method:

- When both eigenvalues are large and approximately equal, the gradients vary significantly in multiple directions. This suggests a well-conditioned system, allowing for reliable estimation of both $u$ and $v$.
- If one eigenvalue is much smaller than the other, it indicates that the gradients are primarily in one direction. This leads to the **aperture problem**, where motion orthogonal to the dominant gradient cannot be reliably determined.

In cases where both eigenvalues are small, it implies insufficient variation in gradients, making the flow estimation unreliable.
