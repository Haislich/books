# Monocular Depth Estimation Techniques

1. **Supervised Learning**:
   - **Supervised learning** is the most straightforward approach, where the model is trained using pairs of images and their corresponding ground truth depth maps. This method requires large amounts of **annotated data** with depth information, which can be expensive and time-consuming to collect.
   - An example dataset used in supervised learning is the **KITTI dataset**, which includes depth maps obtained using LiDAR sensors. KITTI is widely used in autonomous driving research and provides valuable ground truth data for depth estimation.

   **Advantages**:
   - High accuracy when provided with large, high-quality datasets.
   - Direct supervision helps the model learn complex scene structures and depth cues effectively.

   **Disadvantages**:
   - Requires large amounts of labeled data, which may not always be available.
   - Performance can degrade in novel or unseen environments that differ from the training data.

2. **Self-Supervised Learning**:
   - Self-supervised methods have been developed to overcome the dependency on large amounts of ground truth depth data. In **self-supervised monocular depth estimation**, the model learns depth by exploiting geometric constraints and visual consistency between **consecutive video frames** or **stereo images**.
   - For example, in the case of video input, the model predicts depth and uses it to warp the next frame in the sequence. The network is then trained to minimize the difference between the warped frame and the actual next frame, ensuring that the predicted depth is geometrically consistent with the scene's motion.

   **Advantages**:
   - Does not require ground truth depth data, making it more scalable and adaptable to different environments.
   - The model can learn depth information in real-time from video sequences, making it suitable for applications like autonomous driving or robotics.

   **Disadvantages**:
   - Self-supervised methods can struggle with complex scenarios, such as dynamic objects or scenes with large occlusions.
   - Depth accuracy may not be as high as in supervised methods, particularly for scenes with little camera motion or few distinctive features.

3. **Transformer-based Models**:
   - **Transformers** have recently been applied to monocular depth estimation tasks due to their ability to model **long-range dependencies** in the image. Transformers are better than CNNs at capturing global context, which is crucial for understanding the depth of distant or less defined objects in an image.
   - **DepthFormer** and **DepthAny** are examples of Transformer-based depth estimation models. They use the **self-attention mechanism** to analyze relationships between different parts of the image, improving depth prediction in areas that are not well-defined by local features (e.g., distant horizons or highly occluded regions).
   - These models often combine both local and global information, using **hybrid CNN-Transformer architectures**, where CNNs capture local details and Transformers model global context.

   **Advantages**:
   - Better global understanding of the scene, which can improve depth estimation in complex environments.
   - Transformers allow the model to handle large receptive fields, making them effective for capturing both local and distant depth information.

   **Disadvantages**:
   - Higher computational cost compared to pure CNN-based models, due to the heavy use of self-attention mechanisms.
   - Training these models requires large amounts of data and computational resources.
