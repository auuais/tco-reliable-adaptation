"""Hash and archive private evidence on the second disk and attest compact public inputs."""
import argparse
from pathlib import Path
import shutil

from tco_final_ln_protocol import MANIFEST, PREP, PUBLIC, RAW, STUDY, read, seal, sha


SECOND = Path("E:/ResearchData/TCO/research_results")/STUDY


def snapshot(label):
    for terminal in RAW.glob("cases/*/attempt*/start.json"):
        if not (terminal.parent/"terminal.json").is_file():
            raise ValueError("Active or unresolved case: "+str(terminal.parent))
    inventory = {}
    for folder,kind in ((RAW,"raw"),(PREP,"inputs")):
        for source in sorted(folder.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts or source.suffix in (".tmp",".lock",".pyc"):
                continue
            rel = source.relative_to(folder)
            key = kind+"/"+rel.as_posix()
            target = SECOND/kind/rel
            # Include a local C: copy of immutable prepared packets, whose source is already on E:.
            mirrors = [target]
            if kind == "inputs":
                mirrors.append(RAW.parent/(STUDY+"_input_backup")/rel)
            digest = sha(source)
            for mirror in mirrors:
                mirror.parent.mkdir(parents=True, exist_ok=True)
                if not mirror.exists():
                    shutil.copy2(source,mirror)
                if sha(mirror) != digest:
                    raise ValueError("Archive collision or failed verification: "+str(mirror))
            inventory[key] = dict(bytes=source.stat().st_size,sha256=digest)
    manifest = SECOND/(label+"_sha256.json")
    seal(manifest,inventory)
    seal(RAW.parent/(STUDY+"_input_backup")/(label+"_sha256.json"),inventory)
    result = dict(study=STUDY,snapshot=label,files=len(inventory),bytes=sum(item["bytes"] for item in inventory.values()),
                  inventory_sha256=sha(manifest),registration_sha256=sha(MANIFEST),two_disks_verified=True,
                  published_dense_arrays=False,rental_compute_usd=0)
    seal(PUBLIC/(label+"_archive_receipt.json"),result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label",choices=("final",))
    print(snapshot(parser.parse_args().label))
