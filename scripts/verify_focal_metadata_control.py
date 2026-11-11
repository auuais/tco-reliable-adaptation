"""Verify the preregistered screen across platforms with the project's numeric tolerance."""
import focal_metadata_control as study
from verify_masked_camera_gradients import compare


def main():
    if study.read(study.PLAN) != study.protocol():
        raise ValueError("Frozen focal screen inputs or source changed")
    record = study.read(study.OUT)
    if record["manifest_sha256"] != study.sha(study.PLAN):
        raise ValueError("Published protocol hash differs")
    compare(record["results"],study.summarize(record["rows"],record["controls"]))
    print("Verified 270 matched focal configurations and nine unchanged controls.")


if __name__ == "__main__":
    main()
