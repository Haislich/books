# Challenges in Monocular Depth Estimation

1. **Scene Ambiguity**:
   - One of the major challenges is the inherent ambiguity in interpreting depth from a single image. For instance, objects with similar appearances but different sizes can confuse the model. A small car in the foreground may look similar to a large truck in the background, and both could be perceived at different depths.
   - Common visual illusions, like a long road receding into the distance, also make it difficult for the model to accurately gauge the depth without geometric context or learned priors.

2. **Inconsistent Scales**:
   - Monocular depth estimation models often struggle to predict the **absolute scale** of objects. Since the camera's exact field of view or calibration may not be known, the model tends to predict **relative depth** (the relative distance between objects in the scene) rather than **absolute depth** (the actual distance from the camera).
   - This problem is particularly evident in images with unfamiliar objects or scenes without clear visual references for scale.

3. **Occlusions**:
   - **Occlusions** occur when objects in the scene block parts of other objects. Monocular depth estimation models must handle these occlusions intelligently. If a person stands in front of a tree, for example, the model needs to infer the depth of the hidden parts of the tree based on contextual information from other parts of the image.

4. **Lack of Geometric Constraints**:
   - Unlike stereo depth estimation, monocular depth estimation lacks the geometric constraints provided by multiple viewpoints. Thus, the model must rely purely on learned features to infer depth, which can lead to inaccuracies, especially for novel scenes that were not part of the training data.
