# First Networks

1. **SegNet**:
   - A fully convolutional encoder-decoder architecture where the encoder downsamples the input image to extract features, and the decoder upsamples to restore spatial resolution.
   - SegNet uses **max-pooling indices** to guide the upsampling process. During max-pooling, the locations of the maximum activations are saved, and during upsampling, these indices help place important features in the correct positions.
   - It’s a lightweight and efficient architecture for segmentation, but it lacks trainable parameters in the decoder.

2. **Dilated Convolutions**:
   - Dilated (or atrous) convolutions introduce gaps (dilations) between kernel elements, allowing convolutions to have a larger receptive field without increasing the number of parameters.
   - Networks like **DeepLab** use dilated convolutions to extract both fine details and global context, making it popular for high-resolution segmentation tasks.

3. **U-Net**:
   - Developed for biomedical image segmentation, U-Net introduced a **U-shaped architecture** with skip connections between the encoder and decoder, allowing the network to combine low-level spatial information with high-level semantic information.
   - It has become a standard for segmentation tasks due to its ability to generate accurate predictions with fewer training samples.
