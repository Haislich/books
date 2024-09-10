# DSI with Inter-Scanline Consistency

Inter-scanline consistency enforces continuity in the disparity mapping across consecutive rows or scanlines. This is crucial because disparities tend to change gradually, except at object boundaries or occlusions.

1. **Match Scoring**: Compute similarity or dissimilarity scores (e.g., SAD or NCC) for each pixel in the left scanline compared to the right, recording the results in the DSI.
2. **Path Finding**: Using dynamic programming algorithms like the Viterbi algorithm, find a smooth path through the DSI that minimizes the total dissimilarity score.
3. **Consistency Check**: A left-right consistency check can verify that the disparity obtained from one image matches the other, helping to detect occlusions.
