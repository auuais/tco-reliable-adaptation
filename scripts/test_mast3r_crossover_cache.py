"""Check crossover operands and reverse directions using distinct synthetic sources."""
from pathlib import Path
import tempfile
import unittest

import torch

from mast3r_crossover_cache import directed_paths, materialize, validate_pairs
from plan_mast3r_crossover import sha256


class CrossoverCacheTest(unittest.TestCase):
    def test_all_four_cells_keep_the_assigned_operands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["image_a", "image_b", "image_c"]
            inventory = {}
            for source in range(2):
                folder = root / f"source{source}"
                records = {}
                for index, (forward, matching) in enumerate(directed_paths(names)):
                    value = source * 100 + index * 10
                    for relative in (forward, matching):
                        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
                    tensors = tuple(torch.full((2, 2), float(value + component)) for component in range(4))
                    torch.save(tensors, folder / forward)
                    torch.save(((1., 2., 1), (torch.tensor([[index, source]]),
                                             torch.tensor([[source, index]]), torch.tensor([float(value)]))),
                               folder / matching)
                    records.update({p: sha256(folder / p) for p in (forward, matching)})
                inventory[folder.name] = records
            for geometry in range(2):
                for bundle in range(2):
                    output = root / f"G{geometry}B{bundle}"
                    records = materialize(root / f"source{geometry}", root / f"source{bundle}", output, names, inventory)
                    self.assertEqual(len(records), 6)
                    for index, (forward, matching) in enumerate(directed_paths(names)):
                        actual = torch.load(output / forward, weights_only=True)
                        for component, source in enumerate((geometry, bundle, geometry, bundle)):
                            self.assertTrue(torch.equal(actual[component], torch.full((2, 2), float(source * 100 + index * 10 + component))))
                        self.assertEqual(sha256(output / matching), inventory[f"source{bundle}"][matching])
                    (output / directed_paths(names)[-1][0]).unlink()
                    with self.assertRaisesRegex(ValueError, "ordered-pair inventory"):
                        validate_pairs(output, names)
            with self.assertRaisesRegex(ValueError, "Duplicate view"):
                directed_paths(["image_a", "image_a"])


if __name__ == "__main__":
    unittest.main()
