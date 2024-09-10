# Coarse-to-Fine Flow Estimation

The assumption of small motion is crucial for the effectiveness of traditional optical flow methods like Lucas-Kanade. However, in scenarios with fast-moving objects or large displacements between frames, this assumption can fail, making it difficult for these methods to accurately compute motion. In such cases, a **multi-scale, coarse-to-fine approach** becomes highly valuable.

The coarse-to-fine strategy, also known as the **pyramidal approach**, involves creating a pyramid of images where each level is a down-sampled version of the original images, progressively reducing the resolution. By starting at the coarsest level (smallest image), the method first estimates the optical flow at this low resolution, where the apparent motion between frames is significantly reduced due to the smaller scale.

The steps in this process are as follows:

1. **Build Image Pyramids**:  
   Both the current and next frames are processed to generate several layers of reduced-resolution images. Each layer is a down-sampled version of the previous one, forming an image pyramid.

2. **Initial Flow Estimation**:  
   Begin at the top of the pyramid (the coarsest, smallest images) and estimate the optical flow. At this level, even large motions become manageable because of the reduced image size.

3. **Refine Flow at Each Level**:  
   Use the flow estimate from the coarser level to guide the flow estimation at the next finer level down the pyramid. Typically, this refinement involves up-sampling the flow estimate from the coarser level and using it as an initial guess for the finer level.

4. **Iterate Down to the Finest Level**:  
   Continue refining the flow estimates down the pyramid until reaching the bottom, which corresponds to the original image resolution. At each level, the flow is progressively refined for increased accuracy.

At each level, the estimated flow vectors are scaled appropriately to account for the down-sampling effect. This scaling ensures that when an estimated flow is up-sampled to the next finer level, it represents equivalent motion in the higher resolution space. Optical flow equations are solved at each level using methods like Lucas-Kanade, modified for the scale, or other algorithms suited to the specific resolution.

The coarse-to-fine approach significantly improves the accuracy of optical flow estimation in cases with large motions, as it allows for incremental refinement from a manageable approximation of the motion at lower resolutions.
