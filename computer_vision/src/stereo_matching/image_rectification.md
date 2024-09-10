# Image Rectification

If the images are not aligned, **rectification** transforms the images to align the epipolar lines horizontally, simplifying stereo matching. Rectification involves applying homographies to warp the images so that corresponding points lie on the same horizontal lines. This is achieved by decomposing the essential matrix $E$ into rotation ($R$) and translation ($t$), then computing rectifying homographies.
