"""Verify both local disks for model inputs and experimental evidence at a sealed stage."""
import argparse
from pathlib import Path
import shutil

from tco_two_track_protocol import MANIFEST, PRIVATE, PUBLIC, STUDY, digest, read, seal
from score_tco_final_ln import digest as hash_file

PREP = Path("E:/ResearchData/TCO/preparation")/STUDY
ARCHIVE = Path("E:/ResearchData/TCO/research_results")/STUDY


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
    elif stage in ("alternative", "stress_hybrid"):
        ledger=MANIFEST.with_name(f"{STUDY}_{stage}.json")
        release=read(PUBLIC/f"a_{stage}_rows.json")
        ids={j["id"] for j in read(ledger)["jobs"]}
        if release["plan_sha256"]!=digest(ledger) or ids!={r["id"] for r in release["rows"]}:
            raise ValueError("The scored release must cover this registered stage")
        for row in release['rows']:
            if row['status']!='completed':
                continue
            score=PRIVATE/'scores/first'/row['id']/'score.json'
            if digest(score)!=row['score_sha256']:
                raise ValueError('Primary score changed: '+row['id'])
            if row.get('rescore_sha256') and digest(PRIVATE/'scores/independent'/row['id']/'score.json')!=row['rescore_sha256']:
                raise ValueError('Independent score changed: '+row['id'])
        if stage=="alternative":
            qualification=MANIFEST.with_name(f"{STUDY}_engineering_narrow.json")
            if qualification.is_file():
                ids.update(j["id"] for j in read(qualification)["jobs"])
        roots=[]
        for uid in sorted(ids):
            if not list((PRIVATE/'cases'/uid).glob('attempt*/terminal.json')):
                raise ValueError('Missing terminal case: '+uid)
            for relative in (Path('cases')/uid, Path('scores/first')/uid, Path('scores/independent')/uid):
                if (PRIVATE/relative).is_dir():
                    roots.append((PRIVATE/relative,ARCHIVE/'evidence'/relative,'evidence/'+relative.as_posix()))
    elif stage=="development":
        sources=[read(MANIFEST.with_name(STUDY+"_"+name+".json"))["jobs"]
                 for name in ("development","engineering","engineering_v2")]
        for job in sources[0]:
            paths=sorted((PRIVATE/"cases"/job["id"]).glob("attempt[0-9][0-9]"))
            if not paths or not (paths[-1]/"terminal.json").exists():
                raise ValueError("Development still running: "+job["id"])
        ids=sorted({job["id"] for jobs in sources for job in jobs})
        roots=[(PRIVATE/"cases"/uid, ARCHIVE/"evidence/cases"/uid,"evidence/cases/"+uid)
               for uid in ids if (PRIVATE/"cases"/uid).exists()]
    else:
        roots=[(PRIVATE, ARCHIVE/"evidence","evidence")]
    # The E: preparation packets need C: copies independently of the C: model evidence.
    if stage in ("fusion", "final"):
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
    parser.add_argument("stage",choices=("fusion","development","alternative","stress_hybrid","final"))
    archive(parser.parse_args().stage)
