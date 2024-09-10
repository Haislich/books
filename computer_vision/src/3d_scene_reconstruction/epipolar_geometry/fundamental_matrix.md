# Fundamental Matrix

The **fundamental matrix (F)** is a generalization of the essential matrix for **uncalibrated stereo systems**, where the camera intrinsics are unknown. It relates corresponding points $\mathbf{x}$ and $\mathbf{x}'$ between two images:

$$
\mathbf{x}'^T F \mathbf{x} = 0
$$

When the intrinsic camera parameters $K$ and $K'$ are known, the fundamental matrix can be derived from the essential matrix as:

$$
F = K'^{-T} E K^{-1}
$$
