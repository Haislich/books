# Lucas-Kanade Feature Tracker

The Lucas-Kanade method estimates the motion of selected features by assuming that the optical flow of the brightness pattern in the image window remains constant over short time intervals. It calculates the displacement vector (in the x and y directions) for each feature, minimizing the appearance difference in its neighborhood between consecutive frames.

By using a multi-scale (pyramidal) approach, the tracker can handle large motions, refining these estimates at finer scales. This results in a set of flow vectors for each tracked feature, indicating its movement from one frame to the next.
