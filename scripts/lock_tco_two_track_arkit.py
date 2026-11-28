"""Select six disjoint public ARKitScenes visits after the corresponding locked decision."""
import argparse
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import prepare_tco_final_ln_arkit as arkit
from lock_tco_final_ln_replication import discover, number, previous_exposure, site_groups
from lock_tco_two_track_jobs import attach, write_roster
from tco_two_track_protocol import MANIFEST, PUBLIC, ROOT, STUDY, digest, read, schedule, seal

PREP = Path("E:/ResearchData/TCO/preparation") / STUDY
OLD = ROOT/"results"/"tco_final_global_layernorm_v1"


def split_index(splits):
    by_video = {}
    for entry in splits:
        video, visit, split = number(entry["video_id"]),number(entry["visit_id"]),entry["fold"]
        if video is None or visit is None:
            continue
        if video in by_video and by_video[video] != (visit,split):
            raise ValueError("Publisher assigned one video to multiple visits or splits")
        by_video[video] = visit,split
    return by_video


def compact(raw, group):
    h, w = reversed(raw["raster"])
    return dict(room=raw["scene"],scene=raw["scene"],dataset="arkitscenes",visit=raw["visit"],
                video=raw["video"],site_group=group,split=raw["split"],frames=raw["frames"],
                raster=[h,w],slug=raw["scene"].replace("/","__"),input_sha256=raw["input_sha256"],
                reference_sha256=raw["reference_sha256"],input_file_sha256=raw["model_npz_sha256"],
                reference_file_sha256=raw["reference_npz_sha256"])


def intake(split):
    stage = "teacher" if split == "Training" else "fresh"
    trigger = PUBLIC/("a_distillation_trigger.json" if stage=="teacher" else "nominees.json")
    decision = read(trigger)
    if not (decision["proceed"] if stage=="teacher" else (decision.get("a") or decision.get("b"))):
        raise ValueError("No registered branch has earned this intake")
    target = PUBLIC/("arkit_"+stage+"_intake.json")
    output = PUBLIC/("arkit_"+stage+"_inputs.json")
    if target.exists():
        audit=read(target)
        if audit["trigger_sha256"] != digest(trigger) or len(audit["selected"]) != 6:
            raise ValueError("Prior roster has a different decision or exhausted eligibility")
        rooms=audit["selected"]
    else:
        splits, assets, laser, missing, hashes = discover()
        by_video=split_index(splits)
        available=defaultdict(list)
        for video,(visit,fold) in by_video.items():
            if fold==split:
                available[visit].append(video)
        registered=set(read(MANIFEST)["fresh"]["primary"]["explicit_excluded_visits"])
        # The historical search excludes its own study. Include every delivered ARKit
        # stimulus from that study explicitly and include teacher rows when confirming.
        delivered=[read(p)["scene"] for p in sorted(OLD.glob("arkit_validation_*.json"))]
        delivered.append(read(OLD/"arkit_training_input.json")["scene"])
        teacher=PUBLIC/"arkit_teacher_inputs.json"
        if split=="Validation" and teacher.is_file():
            delivered.extend(read(teacher)["rooms"])
        _, sources=previous_exposure(splits)
        # Local partial prep is restartable. Rows from this study's other cohort
        # still enter via delivered; its own unfinished cohort is not prior exposure.
        sources=[r for r in sources if STUDY not in r["path"]]
        prev={v for r in sources for v in r["visits"]}
        exposed=prev|registered|{r["visit"] for r in delivered}
        seen_rgb=set()
        seen_packets={r["input_sha256"] for r in delivered}
        for row in delivered:
            detail=row
            if "source_sha256" not in detail:
                detail=read(PUBLIC/("arkit_teacher_"+str(row["visit"])+".json"))
            seen_rgb.update(v for k,v in detail["source_sha256"].items() if "/lowres_wide/" in "/"+k)
        sites=site_groups(assets,laser)
        denied={sites.get(v,v) for v in exposed}
        metadata={number(r["video_id"]):number(r["visit_id"]) for r in assets}
        excluded,rooms=[],[]

        def membership(fold,visit,video):
            if fold != split or by_video.get(video) != (visit,fold):
                raise ValueError("Visit/video membership differs from pinned publisher split")

        # The older bridge permits exactly one Training visit. Keep its decoder and
        # geometry tests, and supply this registered study's six-visit membership rule.
        with patch.object(arkit,"fetch_membership",membership):
            for visit in sorted(available):
                group=sites.get(visit,visit)
                if group in denied:
                    excluded.append(dict(visit=visit,site_group=group,reason="prior exposure or duplicate site/scan"))
                    continue
                for video in sorted(available[visit]):
                    if video in missing or metadata.get(video) != visit:
                        excluded.append(dict(visit=visit,video=video,reason="missing publisher trajectory or membership"))
                        continue
                    try:
                        raw=arkit.build(arkit.DATA_ROOT,split,visit,video,PREP)
                    except HTTPError as e:
                        if e.code not in (404,410):
                            raise
                        excluded.append(dict(visit=visit,video=video,reason=f"publisher HTTP {e.code}"))
                        continue
                    except ValueError as e:
                        if str(e) not in {"No five-view window from the registered time/pose rules",
                                         "Missing registered public asset","Insufficient adjacent native LiDAR surface overlap"}:
                            raise
                        excluded.append(dict(visit=visit,video=video,reason=str(e)))
                        continue
                    frame_hashes={v for k,v in raw["source_sha256"].items() if "/lowres_wide/" in "/"+k}
                    if len(frame_hashes)!=5:
                        raise ValueError("Ordered window does not contain five distinct RGB sources")
                    if frame_hashes & seen_rgb or raw["input_sha256"] in seen_packets:
                        excluded.append(dict(visit=visit,video=video,reason="duplicate RGB or model input"))
                        continue
                    seal(PUBLIC/f"arkit_{stage}_{visit}.json",raw)
                    rooms.append(compact(raw,group))
                    seen_packets.add(raw["input_sha256"])
                    seen_rgb.update(frame_hashes)
                    denied.add(group)
                    print("Selected",split,visit,video,flush=True)
                    break
                if len(rooms)==6:
                    break
        seal(target,dict(split=split,trigger_sha256=digest(trigger),source_revision=arkit.REVISION,
                         public_metadata_sha256=hashes,prior_visits=sorted(exposed),exposure_sources=sources,
                         ranked_visit_count=len(available),selected=rooms,exclusions=excluded))
    if len(rooms)!=6:
        raise RuntimeError(f"{len(rooms)} eligible visits after exhaustive intake")
    seal(output,dict(rooms=rooms,split=split,study=STUDY,registration_sha256=digest(MANIFEST),intake_sha256=digest(target)))
    if split=="Training":
        planned=schedule("teacher",[r["room"] for r in rooms],[42],["Full40"],3)
        write_roster("distillation_teachers",attach(planned,rooms,PREP),output,dict(trigger_sha256=digest(trigger)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("split",choices=("Training","Validation"))
    intake(p.parse_args().split)
