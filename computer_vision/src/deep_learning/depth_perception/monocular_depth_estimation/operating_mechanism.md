# Operating Mechanism

1. **Monocular Input**:
   - The input to the model is a **single RGB image** (or grayscale image). This image only contains two-dimensional spatial information (height and width), so the task of the network is to predict the **three-dimensional depth** (z-coordinate) from this limited 2D data. Monocular depth estimation tries to infer the depth of each pixel based on texture, shading, perspective, and object size.

2. **Feature Extraction**:
   - The first step involves using a **Convolutional Neural Network (CNN)** to extract features from the image. CNNs are adept at learning spatial hierarchies from images, which are crucial for identifying visual cues (e.g., texture gradients, occlusions, or perspective) that relate to depth.
   - Layers with **dilated convolutions** or **multi-scale feature extraction** may be used to capture features at different scales, as objects of varying distances can have vastly different sizes in the image.

3. **Depth Regression**:
   - After extracting features, the model uses these to predict a **depth map**, which assigns a depth value to each pixel in the image. This is typically done using a regression network, where the CNN outputs are passed through further layers to generate a continuous depth value.
   - Techniques such as **dilated convolutions** (to capture context at larger scales) or **attention mechanisms** (to focus on relevant parts of the image) can be used to improve the depth estimation quality. The result is a depth map, where brighter pixels represent areas closer to the camera and darker pixels represent areas further away.

4. **Supervision**:
   - During training, the model is supervised using **ground truth depth maps**. These depth maps are typically generated using sensors like **LiDAR** (which uses laser pulses to measure distance) or **stereo cameras** (which use two images to compute depth through triangulation).
   - The model's predicted depth map is compared to the ground truth using loss functions such as **Mean Squared Error (MSE)** to minimize the difference between the predicted and actual depth values.
