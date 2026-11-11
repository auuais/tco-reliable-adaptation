# Does updating depth with the camera improve the corresponding surface?

All checks use the original 15 development targets and the registered depth at matching pixels. Normal error is the unsigned angle in camera coordinates on a common smooth, valid reference stencil. Each target carries equal weight. The camera-only half-prior update preserves its original camera-frame depths and normals to floating-point precision.

## Independent tangent planes

Adding the tangent-plane update to the half-weight camera blend reduced squared log axial depth error by 0.86% and raised angular normal error from 37.51 to 40.26 degrees. Depth error fell 1.69% on Heads and 2.10% on Fire and rose 0.85% on Chess. Normal error rose in every scene. Permuting the same nonzero log-range corrections within each frame gave normal errors of 57.18 to 57.21 degrees across four seeds. Their mean depth losses were also higher than the original placement. The correction location carries information, but these changes do not preserve the original local orientation accuracy. [Protocol](../../docs/TANGENT_PAIRED_GEOMETRY.md) and [per-target errors and shuffle controls](tangent_paired_geometry.json).

## Rendering the connected surface

We triangulated each model depth image, removed triangles crossing depth jumps and cast the updated camera rays against that same predicted mesh. The half-weight render reduced paired normal error from 40.26 degrees for tangent planes to 38.05 degrees. Its squared log-depth error was 8.46% worse than retaining the original depth. Just 3.26% of valid pixels had a rendered range change greater than 5%; they contributed an increase of 9.00% of baseline depth error, while smaller changes reduced it. The registered whole-surface score was 0.96521 and the released score was 0.97845, versus 0.95530 and 0.96032 for the half-weight tangent update. The primary render failed the whole-surface criterion. [Mesh protocol](../../docs/CAMERA_MESH_RENDER.md), [canonical results](camera_mesh_scores.json), [matched-pixel results](camera_mesh_paired.json) and [step-size breakdown](camera_mesh_depth_bands.json).
