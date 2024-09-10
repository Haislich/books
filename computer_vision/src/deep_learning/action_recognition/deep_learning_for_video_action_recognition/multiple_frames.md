# Multiple Frames

To capture temporal information, various strategies are introduced that incorporate multiple frames into the analysis:

- **SingleFrame with Temporal Dimension Information Fusion**: Incorporates time by fusing features extracted from multiple frames over time. This method retains the simplicity of analyzing individual frames while incorporating some time context.
- **Late Fusion**: Extracts features from individual frames and combines them at a later stage (e.g., before classification).
- **Slow Fusion**: Instead of simply combining the frames, slow fusion incrementally incorporates information from frames over time, allowing the model to gradually build temporal awareness across frames.

However, the precise implementation of these fusion strategies varies, and further elaboration on how the temporal information is fused is not always provided in the literature.
