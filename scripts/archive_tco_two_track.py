"""Verify both local disks for model inputs and experimental evidence at a sealed stage."""
import argparse
import hashlib
from pathlib import Path
import shutil

from tco_two_track_protocol import PRIVATE, PUBLIC, STUDY, digest, read, seal

PREP = Path("E:/ResearchData/TCO/preparation")/STUDY
ARCHIVE = Path("E:/ResearchData/TCO/research_results")/STUDY


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def copy_checked(src, dest):
    expected=hash_file(src)
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        part=dest.with_name(dest.name+".partial")
        shutil.copyfile(src,part)
        if hash_file(part)!=expected:
            raise ValueError("Incomplete copy: "+str(src))
        part.replace(dest)
    if dest.stat().st_size!=src.stat().st_size or hash_file(dest)!=expected:
        raise ValueError("Conflicting immutable backup: "+str(dest))
    return dict(bytes=src.stat().st_size,sha256=expected)


def archive(stage):
    if stage=="fusion":
        roots=[(PRIVATE/"fusion", ARCHIVE/"evidence/fusion","evidence/fusion")]
    else:
        roots=[(PRIVATE, ARCHIVE/"evidence","evidence")]
    # The E: preparation packets need C: copies independently of the C: model evidence.
    roots.append((PREP,PRIVATE/"input_backup","inputs"))
    records={}
    for source, destination, label in roots:
        if source.drive.lower()==destination.drive.lower():
            raise ValueError("Expected distinct physical volumes")
        for src in sorted(source.rglob("*")):
            rel=src.relative_to(source)
            if not src.is_file() or rel.parts[0] in ("input_backup","archives") or src.name.endswith((".partial",".lock")):
                continue
            if src.is_symlink():
                raise ValueError("Unresolved evidence symlink")
            records[label+"/"+rel.as_posix()]=copy_checked(src,destination/rel)
    index=ARCHIVE/"snapshots"/(stage+".json")
    seal(index,dict(study=STUDY,stage=stage,files=records))
    local=PRIVATE/"archives"/index.name
    copy_checked(index,local)
    public=PUBLIC/("archive_"+stage+".json")
    seal(public,dict(stage=stage,files=len(records),verified_bytes=sum(v["bytes"] for v in records.values()),
                     manifest_sha256=digest(index),two_volumes=True,rental_compute_usd=0))
    print(public,read(public)["verified_bytes"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fusion","final"))
    archive(parser.parse_args().stage)
