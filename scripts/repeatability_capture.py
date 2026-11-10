"""Observation-only capture of predictions, adapters and RNG state; no reseeding."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch


def array_digest(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(key.encode()); digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode()); digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


class Capture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.records = {}
        self.seconds = 0.

    def save_index(self):
        (self.directory / "capture.json").write_text(json.dumps(dict(
            records=self.records, observation_seconds=self.seconds,
            dtype_note="bfloat16 tensors exported as exact float32 values; original dtypes recorded"),
            indent=2, allow_nan=False)+"\n")

    def rng(self, label):
        started = time.perf_counter()
        state = np.random.get_state()
        values = dict(python=random.getstate(), numpy=[state[0],state[1].tolist(),*state[2:]],
                      torch_cpu=torch.get_rng_state().tolist(),
                      torch_cuda=[s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
        raw = (json.dumps(values, sort_keys=True, separators=(",", ":"))+"\n").encode()
        path = self.directory / (label+"_rng.json")
        path.write_bytes(raw)
        self.records[label+"_rng"] = dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest())
        self.seconds += time.perf_counter()-started
        self.save_index()

    def tensors(self, label, values):
        started = time.perf_counter()
        tensors = {key: value for key, value in values.items() if torch.is_tensor(value)}
        arrays = {key: value.detach().cpu().float().numpy() if value.dtype == torch.bfloat16
                  else value.detach().cpu().numpy() for key, value in tensors.items()}
        path = self.directory / (label+".npz")
        assert not path.exists()
        np.savez_compressed(path, **arrays)
        self.records[label] = dict(file=path.name, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  tensor_sha256=array_digest(arrays),
                                  dtypes={key: str(value.dtype) for key, value in tensors.items()},
                                  shapes={key: list(value.shape) for key, value in tensors.items()})
        self.seconds += time.perf_counter()-started
        self.save_index()

    def adapters(self, label, model):
        values = {name: value for name, value in model.named_parameters() if "lora_A" in name or "lora_B" in name}
        assert values, "Missing adapters"
        self.tensors(label, values)
