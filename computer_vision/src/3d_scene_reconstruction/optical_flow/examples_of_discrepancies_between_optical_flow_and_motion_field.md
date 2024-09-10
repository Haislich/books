# Examples of Discrepancies Between Optical Flow and Motion Field

1. **Lambertian Motion Sphere**:  
   Imagine a perfectly smooth, Lambertian sphere (which reflects light diffusely) rotating in space. The sphere's surface points are moving, hence there is a real motion field. However, if the sphere is uniformly colored and the lighting is even, there might be no change in brightness patterns detectable by an observer. Therefore, no optical flow would be observed, even though there is an actual motion field.

2. **Moving Light Around a Stationary Ball**:  
   If the ball itself is stationary, there is no actual motion of the ball's surface points, and thus, the motion field is null. However, as the light moves, it creates changing shadows and highlights on the ball's surface. These changes in brightness are captured as optical flow, indicating apparent motion where there is no actual physical movement of the object.

3. **Barber Pole Illusion**:  
   The actual motion of the stripes on a barber pole is horizontal as the pole rotates around its axis. Visually, due to the cylindrical shape and the observer's usual frontal perspective, the stripes appear to move vertically. This creates an optical flow that is perpendicular to the actual direction of the motion field.
