"""Bind registered model cases to portable files before any model can run."""
from pathlib import Path

from tco_two_track_protocol import MANIFEST, PUBLIC, STUDY, digest, read, seal


def attach(specs, records, prepared):
    rooms = {r["room"]: r for r in records}
    checked = set()
    out = []
    for spec in specs:
        room = rooms[spec["room"]]
        for k in ("frames", "dataset", "input_sha256", "reference_sha256"):
            if spec.get(k, room[k]) != room[k]:
                raise ValueError("Changed case input: " + spec["id"])
        if room["room"] not in checked:
            for role, key in (("model", "input_file_sha256"), ("reference", "reference_file_sha256")):
                if digest(prepared / role / (room["slug"] + ".npz")) != room[key]:
                    raise ValueError("Packet file differs: " + room["room"])
            checked.add(room["room"])
        out.append({**room, **spec, "profile": spec.get("profile", "clean")})
    return out


def write_roster(stage, jobs, source, extra=None):
    if len({j["id"] for j in jobs}) != len(jobs) or not jobs:
        raise ValueError("Invalid job identifiers")
    path = MANIFEST.with_name(STUDY+"_"+stage+".json")
    seal(path, dict(study=STUDY, stage=stage, registration_sha256=digest(MANIFEST),
                    packet_receipt_sha256=digest(source), jobs=jobs, **(extra or {})))
    print(stage, len(jobs), digest(path))
    return path


def main():
    registry = read(MANIFEST)["a"]
    source = PUBLIC / "development_inputs.json"
    rooms = read(source)["rooms"]
    prep=Path("E:/ResearchData/TCO/preparation") / STUDY
    eng = attach(registry["engineering_jobs"], rooms, prep)
    all_jobs = {j["id"]:j for j in attach(registry["dev_jobs"], rooms, prep)}
    write_roster("engineering", eng, source)
    write_roster("pilot", [all_jobs[i] for i in registry["pilot_ids"]], source)
    write_roster("development", [all_jobs[i] for i in registry["clean_order"]], source)


if __name__=="__main__":
    main()
