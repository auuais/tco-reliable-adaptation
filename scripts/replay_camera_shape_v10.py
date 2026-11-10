"""Replay v10 after verifying Git's LF file differs only in its final line ending."""
import hashlib
from pathlib import Path
import camera_shape_v10 as v10

if __name__=="__main__":
    original=v10.sha
    path=Path(v10.__file__).resolve()
    target=v10.read(v10.ROOT/"manifests/camera_shape_v10.json")["sources"]["scripts/camera_shape_v10.py"]
    def verified_sha(p):
        raw=original(p)
        if Path(p).resolve()!=path or raw==target:return raw
        data=path.read_bytes()
        if b"\r" not in data and data.endswith(b"\n") and hashlib.sha256(data[:-1]+b"\r\n").hexdigest()==target:
            return target
        raise ValueError("Source change beyond recorded final line ending")
    v10.sha=verified_sha
    v10.main()
