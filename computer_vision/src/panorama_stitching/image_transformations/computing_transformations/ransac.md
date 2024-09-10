# RANSAC

When computing transformations between two images, such as a homography between image \( A \) and image \( B \), feature matches are crucial. However, due to detection errors, scene changes, and mismatches, outliers can occur—pairs of points that do not correspond to the same physical point in space. Outliers can severely distort the computed transformation, leading to inaccurate results when using least squares optimization.

**RANSAC (Random Sample Consensus)** is a robust algorithm used to estimate the transformation while mitigating the effects of outliers. RANSAC iteratively tries to find a model that best fits the inliers, ignoring the outliers. The steps are:

1. **Sample Minimum Points**: Randomly select the minimal number of point pairs required to estimate the transformation (e.g., 4 pairs for homography).
2. **Fit Model**: Compute the transformation (e.g., homography) from the selected points.
3. **Evaluate Inliers**: Apply the model to all matches and count the inliers—points that conform to the model within a defined error threshold.
4. **Iterate**: Repeat the process, each time with different random point pairs, to find the transformation with the highest number of inliers.
5. **Select Optimal Model**: Choose the model with the most inliers.
6. **Refine Model**: Recalculate the transformation using only the inliers to improve accuracy.

The number of RANSAC iterations \( N \) needed to achieve a high probability \( p \) of success is determined by:

$$
N = \frac{\log(1 - p)}{\log(1 - (1 - e)^s)}
$$

Where:

- \( s \) is the number of points needed for the model (e.g., 4 for a homography),
- \( e \) is the proportion of outliers,
- \( p \) is the desired probability of finding a valid model.

---

**Pros of RANSAC**:

- **Robustness**: Effectively handles datasets with a high proportion of outliers.
- **Generality**: Can be applied to various model-fitting tasks beyond homographies, such as fundamental matrices and affine transformations.

**Cons of RANSAC**:

- **Computational Cost**: High iteration count may be required when the proportion of outliers is large, leading to slower computations.
- **Failure in Dense Outliers**: If outliers dominate the dataset, RANSAC may struggle to find a valid model.
