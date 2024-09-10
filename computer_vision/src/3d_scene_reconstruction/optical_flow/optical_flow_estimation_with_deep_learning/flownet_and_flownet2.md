# FlowNet and FlowNet2

FlowNet is a pioneering CNN architecture for end-to-end optical flow estimation, where the entire process—from input frames to flow fields—is handled by a single model. It uses an encoder-decoder structure, with the encoder capturing abstract representations and the decoder reconstructing the flow field.

Two variants of FlowNet exist:

- **FlowNetSimple**: Stacks two frames and processes them through a single network to directly predict the optical flow.
- **FlowNetCorr**: Uses two streams for the input frames, combining them with a correlation layer to learn displacement between the images.

**FlowNet2** improves on FlowNet by stacking multiple FlowNets, each refining the flow estimate. It introduces sub-networks to better handle large displacements, enhancing accuracy over the original model.
