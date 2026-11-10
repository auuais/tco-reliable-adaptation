"""Evidence integrity must fail closed before loading numerical payloads."""
from pathlib import Path
import tempfile
import unittest
from evaluation_stability import sha, verify


class EvidenceIntegrity(unittest.TestCase):
    def test_changed_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction.bin"
            path.write_bytes(b"original evidence")
            digest = sha(path)
            verify(path, digest)
            path.write_bytes(b"changed evidence")
            with self.assertRaisesRegex(ValueError, "Evidence hash mismatch"):
                verify(path, digest)


if __name__ == "__main__":
    unittest.main()
