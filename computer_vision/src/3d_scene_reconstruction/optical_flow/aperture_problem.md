# Aperture Problem

The aperture problem arises when motion information is available only within a limited field of view, which is common in scenarios where the camera (or aperture) captures only a small part of a larger scene.

This problem highlights a fundamental ambiguity in motion perception:

- **Limited Visibility**:  
  When viewing motion through a small aperture (literally or figuratively, such as a small window on a larger scene), it becomes challenging to discern the true direction of motion if the visible structure does not contain sufficient variation.

- **Edge Motion**:  
  For instance, if you can only see a straight edge moving, without additional context or texture, you can only detect motion along the direction parallel to the edge. Motion perpendicular to the edge becomes indiscernible because the edge appears the same regardless of its movement along its length.

Each instance of the optical flow equation provides only one constraint for the two unknown components of the motion vector (horizontal and vertical). This leads to multiple possible solutions for the true motion vector.

The optical flow calculation at any point depends on the local gradient of image brightness. In areas where this gradient is unidirectional (such as along an edge), the flow component perpendicular to this gradient remains undetermined, manifesting the aperture problem.
