"""Reuse frozen validation under the corrected, separately named study ID."""
import aggregate_pose_precision as base

if __name__ == "__main__":
    base.STUDY = "pose_precision_v7b_local"
    base.main()
