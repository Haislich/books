# Harris Corner Detector

The **Harris Corner Detector** is a well-known method for detecting corners, regions where there is a significant change in intensity in multiple directions. Corners are points where two or more edges meet, and detecting them is essential for tasks such as image matching and object recognition.

The algorithm is based on the idea that corners can be detected by analyzing how the image brightness changes when shifted slightly in different directions. This can be quantified using the **summed square difference (SSD)** function, which compares image patches before and after a shift. The intensity change is large around corners but small around edges or flat regions.

The SSD function is defined as:

$$
E_{\text{SSD}}(u) = \sum_{i} [I_1(x_i + u) - I_0(x_i)]^2
$$

Where:

- \( u \) is the small shift vector.
- \( I_1(x_i + u) \) is the intensity at location \( x_i + u \).
- \( I_0(x_i) \) is the original intensity at location \( x_i \).

We can enhance this with a spatial weighting function:

$$
E_{\text{wSSD}}(u) = \sum_i w(x_i) [I_1(x_i + u) - I_0(x_i)]^2
$$

Here, \( w(x_i) \) is a Gaussian windowing function that emphasizes central pixels in the image patch.
