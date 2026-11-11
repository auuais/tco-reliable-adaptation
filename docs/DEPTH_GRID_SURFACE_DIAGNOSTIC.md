# Can a correct depth choice change the primary surface metric?

The five depth factors offer a 22.85% reduction in registered corresponding-pixel RMSE when each pixel receives reference supervision. The spending criterion uses nearest-neighbor surface distances after a separate alignment. Evaluate three reference-assisted point maps on that actual metric. Use the same registered reference to make all choices, then evaluate the resulting map against both the registered and the released reference. Invalid registered pixels keep their original depth.

1. At every valid pixel, select the depth candidate with the smallest squared distance to the registered reference after the previously used fixed direct-focal similarity.
2. Make that five-way choice only at pixels where the first eligible source actually proposes a nonidentity depth in the sealed visibility study. Keep identity elsewhere.
3. At the same proposed pixels, choose only between the actual first-source proposal and identity using the same squared error.

These are constructed point maps with access to reference geometry. A large surface improvement would establish that such a map is reachable with the chosen fixed rays. A small improvement would only describe these specific pixel-error selectors, since they do not optimize ICP or nearest-neighbor surface error. The three comparisons distinguish a limitation of the candidate grid, the proposal locations and the proposed direction. They cannot qualify an estimator for MASt3R, held-out evaluation or publication claims about inference.
