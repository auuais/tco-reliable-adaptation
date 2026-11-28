"""Independently rescore the earliest available saved case if a prespecified case failed."""
import argparse

from score_tco_final_ln import check_against, require
from score_tco_two_track_a import launch_scores, locate, rescore_ids
from tco_two_track_protocol import MANIFEST, PRIVATE, PUBLIC, STUDY, digest, read, seal


def run(stage):
    roster=MANIFEST.with_name(STUDY+"_"+stage+".json")
    jobs=read(roster)["jobs"]
    table=PUBLIC/("a_"+stage+"_rows.json")
    rows={r["id"]:r for r in read(table)["rows"]}
    require(set(rows)=={j["id"] for j in jobs},"Incomplete case ledger")
    intended=rescore_ids(jobs)
    groups={}
    for job in jobs:
        groups.setdefault((job["dataset"],job["arm"]),[]).append(job)
    tasks, choices=[],[]
    for (dataset,arm),group in sorted(groups.items()):
        ranked=sorted(group,key=lambda j:(j["room"],j["seed"],j["id"]))
        first=ranked[0]["id"]
        require(first in intended,"Prespecified case mismatch")
        selected=next((j for j in ranked if rows[j["id"]]["status"]=="completed"),None)
        choice=dict(dataset=dataset,arm=arm,intended_id=first,intended_status=rows[first]["status"],
                    selected_id=selected["id"] if selected else None)
        choices.append(choice)
        if selected and selected["id"]!=first:
            path=locate(selected)[2][0]/"model"
            tasks.append(dict(job=selected,case=path.relative_to(PRIVATE/"cases").as_posix()))
    lock=PUBLIC/("a_"+stage+"_rescore_selection.json")
    seal(lock,dict(roster_sha256=digest(roster),choices=choices,
                   rule="Original room/seed/ID order. If the intended prediction does not exist, take the next saved prediction. No geometry values enter the ordering."))
    launch_scores(stage+"_fallback_rescore",tasks,True)
    checked=[]
    for choice in choices:
        uid=choice["selected_id"]
        if uid:
            first=PRIVATE/"scores/first"/uid/"score.json"
            second=PRIVATE/"scores/independent"/uid/"score.json"
            audit=read(second)
            require(audit["first_score_sha256"]==digest(first) and digest(first)==rows[uid]["score_sha256"],
                    "Different first score or projection")
            checked.append(dict(**choice,independent_score_sha256=digest(second),absolute_mm_difference=check_against(audit,read(first))))
        else:
            checked.append(dict(**choice,unavailable="No saved prediction in this registered group"))
    out=PUBLIC/("a_"+stage+"_rescore_coverage.json")
    seal(out,dict(selection_sha256=digest(lock),rows_sha256=digest(table),checks=checked,rental_compute_usd=0))
    print(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    run(parser.parse_args().stage)
