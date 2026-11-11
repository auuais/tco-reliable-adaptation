# Supplied-pose result

With a common supplied trajectory the depth-aware method scored 0.87384, versus 0.93704 for spatial interpolation and 1.07873 for unchanged depth. Lower scores are better. The 6.74% improvement over spatial interpolation was positive in every development scene (chess 3.29%, fire 12.04%, heads 1.80%). On the released reference the corresponding scores were 0.88116 and 0.89803. The five predeclared development checks passed. The target depth remained evaluation-only. The trajectory came from the public benchmark.

With model-estimated poses the same depth correction scored 0.98461, slightly worse than the direct baseline. Consequently this is a result for sparse measured depth with a supplied trajectory on three reused development scenes. The next experiment repeats the equal-input comparisons at shifted sample locations. Transfer, unseen scenes, and independent verification remain pending.

Replay the pose transformations and all reported comparisons using `python scripts/replay_supplied_pose_sparse_depth.py`. The public replay includes all eighteen original grid observations from the model-pose study and twenty-four supplied-pose evaluations across both references in the respective results folders.
