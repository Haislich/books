# Stereo Block Matching

In block matching, small blocks around each pixel are compared across a range of disparities to find the best match. The **disparity** that minimizes the cost is selected as the correct match.

1. Set a disparity range $[0, D]$.
2. For each block in the left image, slide it along the corresponding row in the right image.
3. Calculate the similarity score for each position.
4. Select the disparity with the best score.
5. Apply **left-right consistency** checks to improve reliability.
