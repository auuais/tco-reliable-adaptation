"""Byte-verified, reference-free conversion of completed M3 endpoints.

This IO adapter does not authorize model execution or reference access. Its
caller must audit the completed model attempt and bind this converter in the
prospective evaluation registry. Here the exact supplied endpoint bytes and
converter/runtime identities are checked before decoding. Existing attempts,
including interrupted ones, are never overwritten. No reference data is read.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from profiled_artifacts import resolve_logical
from profiled_m3_points import convert_endpoints, POINT_FIELDS


SOURCES = ("scripts/prepare_profiled_m3_conversions.py", "scripts/profiled_m3_points.py",
           "scripts/profiled_artifacts.py")
SCOPE = "reports/lora_profiled_m3_v1/conversions"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path, text=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def dependencies():
    return dict(numpy=np.__version__, opencv=cv2.__version__)


def configuration(root):
    """Collect identities only; the scientific wrapper must register them first."""
    root = Path(root)
    return dict(schema="profiled_m3_conversion_configuration_v1",
                source_sha256_lf={name: sha(resolve_logical(root, name, must_exist=True), text=True)
                                  for name in SOURCES}, dependencies=dependencies())


def verify_configuration(root, registered):
    require(registered == configuration(root), "Converter source/runtime differs from registration")
    own = SOURCES[0]
    require(registered["source_sha256_lf"][own] == sha(Path(__file__), text=True),
            "The executing converter is not the registered source")


def _descriptor(root, record):
    require(isinstance(record, dict) and set(record) == {"file", "sha256"},
            "Endpoint descriptor requires exact file and SHA256")
    require(isinstance(record["sha256"], str) and re.fullmatch("[0-9a-f]{64}", record["sha256"]) is not None,
            "Invalid endpoint SHA256")
    path = resolve_logical(root, record["file"], must_exist=True)
    require(path.suffix == ".npz", "Endpoint must be an NPZ archive")
    require(sha(path) == record["sha256"], "Endpoint bytes differ: " + record["file"])
    return path


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _identity(root, native, final, registered):
    verify_configuration(root, registered)
    paths = dict(native=_descriptor(root, native), final=_descriptor(root, final))
    require(native != final and native.get("file") != final.get("file"), "Endpoints must use distinct files")
    return paths


def _destination(root, job_id, attempt):
    for label, value in (("job ID", job_id), ("attempt", attempt)):
        require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is not None,
                "Invalid " + label)
    return resolve_logical(root, SCOPE + "/" + job_id + "/" + attempt)


def prepare(root, job_id, attempt, native, final, registered):
    """Create a fresh conversion from audited descriptors, without references.

    The returned receipt proves only these conversions. A job name is a label,
    not evidence of scientific inventory membership or model completion.
    """
    root = Path(root).resolve()
    target = _destination(root, job_id, attempt)
    require(not target.exists(), "Preserve existing conversion attempt; use a fresh path")
    paths = _identity(root, native, final, registered)
    target.mkdir(parents=True, exist_ok=False)
    identity = dict(schema="profiled_m3_conversion_v1", job_id=job_id, attempt=attempt,
                    native=native, final=final, configuration=registered,
                    references_loaded=False, accuracy_scoring_performed=False,
                    model_completion_verified=False, score_access_authorized=False)
    dump_new(target / "started.json", dict(identity, started_at_utc=datetime.now(timezone.utc).isoformat()))
    try:
        converted = convert_endpoints(_load(paths["native"]), _load(paths["final"]))
        outputs = {}
        for endpoint in ("native", "final"):
            require(set(converted[endpoint]) == POINT_FIELDS, "Converted field inventory differs")
            path = target / (endpoint + ".npz")
            with path.open("xb") as stream:
                np.savez_compressed(stream, **converted[endpoint])
            outputs[endpoint] = dict(file=path.relative_to(root).as_posix(), sha256=sha(path))
        _identity(root, native, final, registered)
        record = dict(identity, outputs=outputs, validation=converted["validation"],
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      status="converted_without_references", pass_all=True)
        dump_new(target / "complete.json", record)
        return record
    except Exception as error:
        dump_new(target / "failure.json", dict(identity, error_type=type(error).__name__,
                 message=str(error), failed_at_utc=datetime.now(timezone.utc).isoformat(), pass_all=False))
        raise


def verify(root, job_id, attempt, native, final, registered, *, replay=True):
    """Recompute conversion exactly by default before the first stage seal.

    Byte-only rechecks are available for an already independently pinned receipt;
    they must not be used to first accept self-described output hashes.
    """
    root = Path(root).resolve()
    target = _destination(root, job_id, attempt)
    paths = _identity(root, native, final, registered)
    record = read(target / "complete.json")
    expected = dict(schema="profiled_m3_conversion_v1", job_id=job_id, attempt=attempt,
                    native=native, final=final, configuration=registered,
                    references_loaded=False, accuracy_scoring_performed=False,
                    model_completion_verified=False, score_access_authorized=False,
                    status="converted_without_references", pass_all=True)
    require(all(record.get(key) == value for key, value in expected.items()), "Conversion identity differs")
    require(not (target / "failure.json").exists(), "Failed conversion cannot be accepted")
    require({p.name for p in target.iterdir()} == {"started.json", "complete.json", "native.npz", "final.npz"},
            "Conversion file inventory differs")
    start = read(target / "started.json")
    require(all(start.get(key) == record[key] for key in expected if key not in ("status", "pass_all")),
            "Started conversion identity differs")
    require(set(record.get("outputs", {})) == {"native", "final"}, "Conversion endpoints incomplete")
    for endpoint, descriptor in record["outputs"].items():
        require(descriptor["file"] == (target / (endpoint + ".npz")).relative_to(root).as_posix(),
                "Converted endpoint escaped its reserved attempt")
        _descriptor(root, descriptor)
    if replay:
        converted = convert_endpoints(_load(paths["native"]), _load(paths["final"]))
        require(converted["validation"] == record["validation"], "Recomputed conversion validation differs")
        for endpoint, descriptor in record["outputs"].items():
            actual = _load(resolve_logical(root, descriptor["file"], must_exist=True))
            require(set(actual) == POINT_FIELDS, "Saved converted field inventory differs")
            for name, value in converted[endpoint].items():
                require(actual[name].dtype == value.dtype and np.array_equal(actual[name], value),
                        "Recomputed converted field differs: " + endpoint + "/" + name)
        _identity(root, native, final, registered)
    return dict(file=(target / "complete.json").relative_to(root).as_posix(),
                sha256=sha(target / "complete.json"), outputs=record["outputs"],
                bytes_verified=True, conversion_replayed=replay, score_access_authorized=False)
