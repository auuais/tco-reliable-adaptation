"""Verify branch decisions against analytically specified reconstruction scores."""
import unittest

from plan_mast3r_crossover import MANIFEST, read
from score_mast3r_crossover import summarize


class CrossoverScoringTest(unittest.TestCase):
    def result(self, geometry, observations, replay_offset=0):
        plan = read(MANIFEST)
        scores = {"G0B0": 1., "G1B1": .9, "G1B0": geometry, "G0B1": observations}
        rows, parent_rows, actual, old = [], [], {}, {}
        def metric(value):
            return dict(accuracy_mean_m=value, completeness_mean_m=value, normal_consistency_mean=.7)
        for scene in plan["scenes"]:
            key, digest = scene + "_base", scene + "_ordinary"
            parent_rows.append(dict(tag=digest, scene=scene, arm="zero", post="ordinary", prediction=digest))
            old[key] = {digest: {ref: metric(1.) for ref in plan["references"]}}
            actual[key] = {}
        for job in plan["runs"]:
            row = dict(**job, input_key=job["scene"] + "_base", prediction=job["id"])
            rows.append(row)
            cell = f'G{row["geometry"]}B{row["bundle"]}'
            actual[row["input_key"]][row["prediction"]] = {ref: dict(metrics=metric(scores[cell])) for ref in plan["references"]}
            if "endpoint" in row:
                tag = row["endpoint"]["tag"]
                parent_rows.append(dict(tag=tag, prediction=tag, arm="D", post="KDP_clean", scene=job["scene"]))
                old[row["input_key"]][tag] = {ref: metric(scores[cell] + replay_offset) for ref in plan["references"]}
        return summarize(dict(rows=rows), actual, dict(rows=parent_rows), old)

    def test_geometry_improvement_selects_geometry(self):
        result = self.result(.8, 1.1)
        self.assertEqual(result["selected_branch"], "geometry")
        contrasts = result["conditions"]["centered_clean"]["registered_rgb"]["contrasts"]
        self.assertAlmostEqual(contrasts["geometry_gain"], .2)
        self.assertAlmostEqual(contrasts["interaction"], 0.)

    def test_observation_improvement_selects_observations(self):
        self.assertEqual(self.result(1.1, .8)["selected_branch"], "observations")

    def test_replay_uncertainty_can_prevent_nomination(self):
        self.assertIsNone(self.result(.8, 1.1, replay_offset=.1)["selected_branch"])


if __name__ == "__main__":
    unittest.main()
