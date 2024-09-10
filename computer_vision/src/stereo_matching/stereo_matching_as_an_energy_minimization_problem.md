# Stereo Matching as an Energy Minimization Problem

Stereo matching can be formulated as an energy minimization problem. The goal is to find the disparity for each pixel that minimizes a global energy function, which typically includes terms for both image similarity and smoothness.

The energy function $E(d)$ is a combination of a data term and a smoothness term:

$$
E(d) = E_d(d) + \lambda E_s(d)
$$

Where:

- $E_d(d)$ is the **data term**, measuring the similarity between corresponding pixels in the left and right images.
- $E_s(d)$ is the **smoothness term**, penalizing large disparity differences between neighboring pixels.
- $\lambda$ is a weighting parameter.

The **data term** can be computed as the Sum of Squared Differences (SSD) between pixel intensities in corresponding windows:

$$
E_d(d) = \sum_{(x, y) \in I} C(x, y, d(x, y))
$$

Where $C(x, y, d(x, y))$ is the cost of assigning disparity $d(x, y)$ to pixel $(x, y)$:

$$
C(x, y, d) = \text{SSD}(\text{window centered at } (x, y) \text{ in left image}, \text{window centered at } (x + d, y) \text{ in right image})
$$

The **smoothness term** is:

$$
E_s(d) = \sum_{(p, q) \in \mathcal{E}} V(d_p, d_q)
$$

Where $V(d_p, d_q)$ is a penalty function applied to the disparity values of neighboring pixels $p$ and $q$. A common choice is the L1 norm or the Potts model.

The dynamic programming approach finds disparities for all pixels such that the total energy $E(d)$ across the image is minimized. This is done by recursively calculating:

$$
D(x, y, d) = C(x, y, d) + \min_{d'} \left\{ D(x - 1, y, d') + \lambda \left| d - d' \right| \right\}
$$

Where $D(x, y, d)$ represents the cumulative cost of assigning disparity $d$ to pixel $(x, y)$. The recursion ensures that both the immediate cost and the transition cost (which enforces smoothness) are considered, resulting in a globally optimized disparity map.
