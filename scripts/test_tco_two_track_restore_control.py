"""Check that the timed trained-suffix change leaves the uninitialized control paths identical."""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
# These semantic digests are from 7dd4286, before the out-of-band restore audit.
# Erase only the body of one disabled branch. Preserve its test and its else suite.
CONTROL = {
    "tco_two_track_hybrid.py": ("initialization is not None", "6b5af1957fe1e22dfcbc4cf61a9fd77d835edc0ae4087f6f0e936eefa6cc38ab"),
    "run_tco_two_track_a_local.py": ('job.get("initialization_sha256")', "d2e30f7d5e0d14d6139c7cddc52a00381a1ce427ee63e6c0b738d6cb126bbedc"),
}


def control_digest(source, predicate):
    tree = ast.parse(source)
    exact = ast.dump(ast.parse(predicate, mode="eval").body)
    branches = [node for node in ast.walk(tree) if isinstance(node,ast.If) and ast.dump(node.test)==exact]
    if len(branches)!=1:
        raise ValueError("Expected exactly one guarded restoration path")
    branches[0].body = [ast.Pass()]
    # ast.dump formatting changed across Python versions. Use field names/values
    # directly, ignoring source locations and newly introduced optional fields.
    def semantic(n):
        if isinstance(n,ast.AST):
            return (type(n).__name__, tuple((k,semantic(v)) for k,v in ast.iter_fields(n)
                           if v is not None and v != [] and k not in {"type_params","type_comment"}))
        if isinstance(n,list):
            return tuple(semantic(x) for x in n)
        return n
    return hashlib.sha256(repr(semantic(tree)).encode("utf-8")).hexdigest()


class ControlPathTest(unittest.TestCase):
    def test_policy_and_launcher(self):
        for name,(predicate,expected) in CONTROL.items():
            with self.subTest(name=name):
                self.assertEqual(control_digest((ROOT/name).read_text(),predicate),expected)


if __name__=="__main__":
    unittest.main()
