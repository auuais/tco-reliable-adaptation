"""Exercise the held-source release path using generated measurements only."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import score_profiled_compensation as scoring
import verify_compensation_public_scores as independent
from replay_compensation_published_metrics import replay
from run_profiled_d0 import read, sha, write_new


def write_inputs(root):
    raw = root / "results/mast3r_partner_shape"
    raw.mkdir(parents=True)
    split_dir = root / "results/lora_profiled_v1"
    split_dir.mkdir(parents=True)
    # The shared pixel has intentionally wrong depth. The fixed mask must exclude it.
    rows = {
        "earlier": np.array([[0., 0., 20.], [2., 3., 2.]]),
        "later": np.array([[0., 0., 30.], [5., 7., 2.]]),
    }
    packets = {}
    splits = {}
    for side in scoring.SIDES:
        path = raw / (side + ".npz")
        keys = [f"{scene}_{frame}" for scene in scoring.SCENES for frame in scoring.FRAMES]
        np.savez_compressed(path, **{key: rows[side] for key in keys})
        packets[side] = sha(path)
        xy, _ = scoring.native_rows(rows[side])
        splits.update({f"{scene}_{side}_{frame}__xy": xy for scene in scoring.SCENES
                       for frame in scoring.FRAMES})
    np.savez_compressed(split_dir/"splits.npz", **splits)
    write_new(split_dir/"inputs.json", dict(splits_binary_sha256=sha(split_dir/"splits.npz"),
                                          registration=dict(packet_sha256=packets)))


def write_predictions(root):
    prediction_dir = root / "predictions"
    prediction_dir.mkdir()
    depth = np.full((5,384,512), .5)
    path = prediction_dir / "native.npz"
    np.savez_compressed(path, depth=depth)
    bound = dict(native_file=path.relative_to(root).as_posix(), native_sha256=sha(path), seconds=1.)
    outcomes = {}
    for name, scale in (("control", 1.), ("pass", 1.02), ("weak", 1.005)):
        path = prediction_dir / f"{name}.npz"
        np.savez_compressed(path, g0=2., scales=np.full(5, scale), depth=depth * (2.*scale))
        outcomes[name] = dict(bound, final_file=path.relative_to(root).as_posix(), final_sha256=sha(path))
    return outcomes


class VerdictPipelineTests(unittest.TestCase):
    def exercise(self, root, *, later_candidate):
        def descriptor(name):
            return dict(file=name, sha256=sha(root / name))

        write_inputs(root)
        outcomes = write_predictions(root)
        cases = []
        for scene in scoring.SCENES:
            for side in scoring.SIDES:
                for arm in scoring.ARMS:
                    choice = "control" if arm != "R1" else ("pass" if side == "earlier" else later_candidate)
                    cases.append(dict(scene=scene, side=side, arm=arm, run_id=f"{scene}_{side}_{arm}",
                                      selected=outcomes[choice]))
        seal = dict(cases=cases, completed_states=36, pass_all=True, numerical_failures=0)
        write_new(root / scoring.SEAL, seal)
        write_new(root / scoring.EVALUATION, dict(synthetic_fixture=True))
        independent_receipt = root / scoring.PUBLIC / "independent_replay.json"
        with (patch.object(scoring, "ROOT", root),
              patch.object(scoring, "file_identity", descriptor),
              patch.object(independent, "ROOT", root),
              patch("register_profiled_compensation.verify_release", return_value=({}, {})),
              patch("register_profiled_compensation.require_scoring_runtime"),
              patch.object(scoring, "build_seal", return_value=dict(seal, completed_states=35)) as recompute,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(ValueError, "seal does not reproduce"):
                scoring.score()
            self.assertFalse((root / scoring.ACCESS).exists())
            recompute.return_value = seal

            scoring.score()
            scoring.score(replay=True)
            independent.verify(output=independent_receipt)

        scored = read(root / scoring.SCORES)
        self.assertEqual(len(scored["rows"]), 180)
        self.assertTrue(all(row["held_count"] == 1 and row["overlapping_excluded"] == 1
                            for row in scored["rows"]))
        self.assertTrue(all(row["native"]["mae_m"] == 1. for row in scored["rows"]))
        self.assertEqual(read(independent_receipt)["report_sha256"], sha(root / scoring.SCORES))
        self.assertEqual(read(root / scoring.PUBLIC / "source_replay.json")["sample_replay"],
                         scored["sample_replay"])
        nomination = replay(scored)["nomination"]
        self.assertEqual(read(independent_receipt)["nomination"], nomination)
        return nomination

    def test_source_replay_and_public_nomination(self):
        with tempfile.TemporaryDirectory(prefix="compensation-verdict-") as d:
            self.assertEqual(self.exercise(Path(d), later_candidate="pass"), "R1")

    def test_source_replay_and_public_rejection_for_one_weak_direction(self):
        with tempfile.TemporaryDirectory(prefix="compensation-verdict-") as d:
            self.assertIsNone(self.exercise(Path(d), later_candidate="weak"))


if __name__ == "__main__":
    unittest.main()
