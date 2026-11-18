"""Snapshot completed profiled-LoRA evidence onto a second physical disk.

Targets are explicit: --m1 selects completed engineering proofs and their exact
endpoint caches; --seal selects the complete 36-state prediction inventory.
No model, score calculation, array deserialization, deletion, or mutable
progress/log collection occurs. A snapshot is valid only after manifest.json
is published following copy, source-stability and destination-hash checks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import traceback

from profiled_artifacts import logical_inventory, resolve_logical


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = "results/lora_profiled_v1"
RAW = "reports/lora_profiled_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path, text=False):
    if text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def dump_new(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")


class Selection:
    """An explicit, deduplicated set of immutable logical filenames."""
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.files={}
        self.guards={}
        self.targets=[]

    def add(self,name,expected=None,text_expected=None):
        path=resolve_logical(self.root,name,must_exist=True)
        require(path.name != "progress.json" and path.suffix.lower() != ".log" and "logs" not in Path(name).parts,
                f"Mutable progress/log files are excluded from immutable archives: {name}")
        before=path.stat()
        digest=sha(path)
        after=path.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),f"Source changed while hashing: {name}")
        if expected is not None:
            require(digest==expected,f"Source hash differs from its completion proof: {name}")
        if text_expected is not None:
            require(sha(path,text=True)==text_expected,f"Source text differs from its registered version: {name}")
        row=dict(file=name,bytes=after.st_size,sha256=digest)
        if name in self.files:
            require(self.files[name]==row,f"Conflicting source identities: {name}")
        self.files[name]=row
        return path

    def marker(self,name,expected=None):
        path=self.add(name,expected)
        self.guards[name]=self.files[name]["sha256"]
        return read(path)

    def tree(self,name,expected):
        actual=logical_inventory(resolve_logical(self.root,name))
        require(set(actual)==set(expected),f"Completed tree has missing or unexpected files: {name}")
        for relative,digest in expected.items():
            self.add(name+"/"+relative,digest)

    def provenance(self,record):
        for name,digest in record.get("input_sha256",{}).items():
            self.add(name,digest)
        for name,digest in record.get("artifact_sha256",{}).items():
            self.add(name,digest)
        for name,digest in record.get("source_sha256_lf",{}).items():
            self.add(name,text_expected=digest)

    def finish(self):
        rows=[self.files[name] for name in sorted(self.files)]
        return dict(schema="profiled_archive_selection_v1",targets=self.targets,
                    completion_guards=self.guards,files=rows,total_bytes=sum(row["bytes"] for row in rows),
                    mutable_progress_or_logs_included=False,arrays_deserialized=False,scores_computed=False)


def no_active_target(root,run_ids):
    lock=resolve_logical(root,RAW+"/scientific_gpu.lock")
    if lock.exists():
        owner=read(lock)
        require(owner.get("run_id") not in run_ids,
                "A selected scientific case still owns the active GPU lock; archive after it finishes")


def collect_m1(selection):
    root=selection.root
    endpoint_name=PUBLIC+"/endpoint_bridge.json"
    gradient_name=PUBLIC+"/gradient_bridge.json"
    endpoint=selection.marker(endpoint_name)
    gradient=selection.marker(gradient_name)
    require(endpoint.get("schema")=="profiled_endpoint_bridge_v1" and all(endpoint.get(k) is True for k in
            ("pass_all","original_state_unchanged","native_endpoint_pass","final_endpoint_pass","adapter_state_unchanged")),
            "M1 endpoint bridge is incomplete or failed")
    require(all(gradient.get(k) is True for k in
            ("pass_all","original_state_unchanged","zero_adapter_pass","replay_gradient_pass","finite_difference_pass","full_macro_step_pass")),
            "M1 gradient bridge is incomplete or failed")
    attempt=endpoint.get("attempt")
    require(isinstance(attempt,str) and re.fullmatch(r"[A-Za-z0-9_-]+",attempt),"Invalid endpoint attempt identity")
    public=PUBLIC+"/endpoint_attempts/"+attempt
    raw=RAW+"/endpoint_attempts/"+attempt
    require(not resolve_logical(root,public+"/failure.json").exists(),"Endpoint attempt has a preserved failure")
    selection.marker(public+"/comparison.json",selection.files[endpoint_name]["sha256"])
    manifest_name=endpoint["engineering_manifest"]
    manifest_hash=endpoint["engineering_manifest_sha256"]
    require(gradient["engineering_manifest_sha256"]==manifest_hash,"M1 bridges use different engineering registries")
    manifest=selection.marker(manifest_name,manifest_hash)
    # Preserve the historical manifest and its bound inputs. A newer registry
    # implementation need not reproduce its old source inventory to archive it.
    for name,digest in manifest.get("artifact_sha256",{}).items():
        selection.add(name,digest)
    selection.provenance(endpoint)
    selection.provenance(gradient)
    require(set(endpoint["captures"])=={"pristine","zero_lora"},"M1 bridge must contain both endpoint variants")
    expected_public={"comparison.json":selection.files[endpoint_name]["sha256"]}
    expected_raw={}
    for label,folder in (("pristine","pristine"),("zero_lora","zero")):
        capture=endpoint["captures"][label]
        for kind in ("native","final"):
            name=raw+"/"+folder+"/"+kind+".npz"
            require(capture[kind+"_file"]==name,"Endpoint capture points outside its completed attempt")
            selection.add(name,capture[kind+"_sha256"])
            expected_raw[folder+"/"+kind+".npz"]=capture[kind+"_sha256"]
        graph_name=public+"/"+folder+"/native_graph.json"
        require(capture["graph_file"]==graph_name,"Endpoint graph points outside its completed attempt")
        graph=selection.marker(graph_name,capture["graph_sha256"])
        cache_name=raw+"/"+folder+"_cache"
        selection.tree(cache_name,graph["cache_files"])
        for name,digest in graph["cache_files"].items():
            expected_raw[folder+"_cache/"+name]=digest
        for filename in ("native.json","native_trace.json","native_graph.json","final.json"):
            name=public+"/"+folder+"/"+filename
            selection.add(name)
            expected_public[folder+"/"+filename]=selection.files[name]["sha256"]
        require(read(resolve_logical(root,public+"/"+folder+"/native.json"))==capture["capture"],
                "Native metadata differs from the completed bridge")
        require(read(resolve_logical(root,public+"/"+folder+"/final.json"))==capture["final"],
                "Final metadata differs from the completed bridge")
    selection.tree(public,expected_public)
    selection.tree(raw,expected_raw)
    gradient_attempt=gradient["attempt"]
    require(isinstance(gradient_attempt,str) and gradient_attempt.startswith(PUBLIC+"/engineering/"),
            "Gradient attempt is outside the engineering evidence directory")
    require(not resolve_logical(root,gradient_attempt+"/failure.json").exists(),"Gradient attempt has a preserved failure")
    selection.marker(gradient_attempt+"/report.json",selection.files[gradient_name]["sha256"])
    finite=selection.marker(gradient_attempt+"/finite_differences.json")
    require(finite.get("pass_all") is True,"Gradient finite-difference evidence did not pass")
    selection.tree(gradient_attempt,{"report.json":selection.files[gradient_name]["sha256"],
        "finite_differences.json":selection.files[gradient_attempt+"/finite_differences.json"]["sha256"]})
    extra=PUBLIC+"/final_calibration_checks.json"
    require(selection.marker(extra).get("pass_all") is True,"Final calibration CPU proof did not pass")
    selection.targets.append(dict(kind="M1",endpoint_attempt=attempt,gradient_attempt=gradient_attempt,
                                   engineering_manifest=manifest_name,completed=True))


def collect_m2(selection,seal_name):
    # This verifier is byte/JSON only. It imports neither a model nor NumPy and
    # does not decode any observation, adapter, or prediction arrays.
    from seal_profiled_predictions import verify_seal

    seal=selection.marker(seal_name)
    require(seal.get("completed_states")==seal.get("required_states")==36 and seal.get("pass_all") is True,
            "M2 requires an already-complete 36-state global prediction seal")
    no_active_target(selection.root,{case["run_id"] for case in seal["cases"]})
    verified=verify_seal(selection.root,seal_name,seal["evaluation_execution"])
    require(verified==seal,"M2 global prediction seal changed")
    manifest=selection.marker(seal["common_scientific_registry"],seal["common_scientific_registry_sha256"])
    selection.provenance(manifest)
    evaluation=selection.marker(seal["evaluation_execution"]["file"],seal["evaluation_execution"]["sha256"])
    selection.provenance(evaluation)
    for name,digest in evaluation.get("packet_sha256",{}).items():
        selection.add(name,digest)
    if "fixtures" in evaluation:
        selection.add(evaluation["fixtures"],evaluation["fixtures_sha256"])
    selection.add(seal["evaluation_rules"],seal["evaluation_rules_sha256"])
    lookup={job["run_id"]:job for job in manifest["jobs"]}
    for case in seal["cases"]:
        job=lookup[case["run_id"]]
        for attempt in case["attempts"]:
            name=attempt["attempt"]
            selection.marker(attempt["report"],attempt["report_sha256"])
            selection.tree(job["public_attempt_parent"]+"/"+name,attempt["public_files"])
            selection.tree(job["raw_attempt_parent"]+"/"+name,attempt["raw_files"])
    selection.targets.append(dict(kind="M2",global_seal=seal_name,completed_states=36,
                                   failure_review_required=seal.get("failure_review_required",False),completed=True))


def collect_selection(root,*,m1=False,seal=None):
    require(m1 or seal,"Choose an explicit --m1 and/or completed --seal target")
    selection=Selection(root)
    if m1:
        collect_m1(selection)
    if seal:
        collect_m2(selection,seal)
    for name in ("scripts/archive_profiled_evidence.py","scripts/profiled_artifacts.py"):
        selection.add(name)
    return selection.finish()


def physical_disks(root,destination):
    require(os.name=="nt","Production archival requires the verified Windows C:/E: physical-disk check")
    result={}
    for label,path in (("source",root),("destination",destination)):
        drive=path.resolve().drive.rstrip(":")
        require(re.fullmatch(r"[A-Za-z]",drive) is not None,"Expected an explicit local Windows drive")
        command=f"(Get-Partition -DriveLetter '{drive}').DiskNumber"
        number=subprocess.check_output(["powershell.exe","-NoProfile","-Command",command],text=True).strip()
        require(number.isdigit(),"Could not establish physical disk identity")
        result[label]=dict(drive=drive.upper(),disk_number=int(number))
    require(result["source"]["disk_number"]!=result["destination"]["disk_number"],"Backup must be on a different physical disk")
    return result


def snapshot_path(root,destination,name):
    root,destination=Path(root).resolve(),Path(destination).resolve()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}",name) is not None,"Invalid fresh snapshot ID")
    require(not destination.is_relative_to(root) and not root.is_relative_to(destination),
            "Archive and repository roots must not contain one another")
    return destination/"snapshots"/name


def copy_file(source,target,record):
    before=source.stat()
    require(before.st_size==record["bytes"],"Source size changed before copy")
    target.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256()
    with source.open("rb") as input_stream, target.open("xb") as output_stream:
        for block in iter(lambda:input_stream.read(1024*1024),b""):
            output_stream.write(block)
            digest.update(block)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    after=source.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),"Source was actively modified during copy")
    require(digest.hexdigest()==record["sha256"] and sha(source)==record["sha256"],"Source bytes changed during copy")
    require(target.stat().st_size==record["bytes"] and sha(target)==record["sha256"],"Destination copy hash/size mismatch")


def archive_snapshot(root,destination,name,*,m1=False,seal=None,disk_probe=physical_disks,after_copy=None):
    """New immutable snapshot; disk_probe/after_copy injections are fixture-only."""
    root,destination=Path(root).resolve(),Path(destination).resolve()
    target=snapshot_path(root,destination,name)
    require(not target.exists(),"Preserve existing completed or failed snapshots; choose a fresh ID")
    disks=disk_probe(root,destination)
    plan=collect_selection(root,m1=m1,seal=seal)
    ancestor=destination
    while not ancestor.exists():
        ancestor=ancestor.parent
    require(shutil.disk_usage(ancestor).free>=plan["total_bytes"]+16*1024**2,"Insufficient archive disk space")
    target.mkdir(parents=True,exist_ok=False)
    data=target/"data"
    data.mkdir()
    started=datetime.now(timezone.utc).isoformat()
    intent=dict(schema="profiled_archive_intent_v1",snapshot=name,source_root=str(root),
                archive_root=str(destination),started_at_utc=started,physical_disks=disks,selection=plan)
    dump_new(target/"intent.json",intent)
    copied=0
    try:
        for row in plan["files"]:
            source=resolve_logical(root,row["file"],must_exist=True)
            output=resolve_logical(data,row["file"])
            copy_file(source,output,row)
            copied+=1
            if after_copy is not None:
                after_copy(copied,row)
        require(collect_selection(root,m1=m1,seal=seal)==plan,
                "Completion markers or source inventory changed while archiving; snapshot is not complete")
        expected={row["file"]:row for row in plan["files"]}
        actual=logical_inventory(data)
        require(set(actual)==set(expected),"Archive destination inventory differs")
        require(all(path.stat().st_size==expected[name]["bytes"] and sha(path)==expected[name]["sha256"]
                    for name,path in actual.items()),"Final destination verification failed")
        report=dict(schema="profiled_evidence_archive_v1",snapshot=name,source_root=str(root),
            archive_root=str(destination),started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
            physical_disks=disks,selection=plan,intent_sha256=sha(target/"intent.json"),
            files_verified=copied,total_bytes=plan["total_bytes"],source_rechecked_after_copy=True,
            destination_hashes_verified=True,logical_filename_codec="Windows U+F03A for Linux ':' only; bytes unchanged",
            arrays_deserialized=False,scores_computed=False,files_deleted=0,pass_all=True)
        dump_new(target/"manifest.json",report)
        return report
    except Exception as error:
        dump_new(target/"failure.json",dict(snapshot=name,error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),files_copied_before_failure=copied,pass_all=False,
            recovery="Preserve all partial bytes. Use a new snapshot ID after investigating; this utility never deletes or overwrites."))
        raise


def verify_snapshot(destination,name,*,source_root=None):
    destination=Path(destination).resolve()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}",name) is not None,"Invalid snapshot ID")
    target=destination/"snapshots"/name
    require(not (target/"failure.json").exists(),"Snapshot is a preserved failed attempt")
    report=read(target/"manifest.json")
    require(report.get("pass_all") is True and report.get("schema")=="profiled_evidence_archive_v1",
            "Snapshot has no valid completion manifest")
    require(sha(target/"intent.json")==report["intent_sha256"],"Archive intent changed")
    intent=read(target/"intent.json")
    require(report.get("snapshot")==intent.get("snapshot")==name and
            report.get("selection")==intent.get("selection") and
            report.get("source_root")==intent.get("source_root") and
            report.get("physical_disks")==intent.get("physical_disks"),
            "Archive completion differs from its original copy intent")
    expected={row["file"]:row for row in report["selection"]["files"]}
    require(len(expected)==report["files_verified"],"Archive manifest has duplicate or missing paths")
    require(sum(row["bytes"] for row in expected.values())==report["total_bytes"],"Archive byte total differs from its inventory")
    actual=logical_inventory(target/"data")
    require(set(actual)==set(expected),"Archived file inventory changed")
    for file_name,path in actual.items():
        require(path.stat().st_size==expected[file_name]["bytes"] and sha(path)==expected[file_name]["sha256"],
                f"Archived bytes changed: {file_name}")
        if source_root is not None:
            source=resolve_logical(source_root,file_name,must_exist=True)
            require(sha(source)==expected[file_name]["sha256"],f"Current source differs from snapshot: {file_name}")
    return dict(snapshot=name,files_verified=len(expected),total_bytes=sum(row["bytes"] for row in expected.values()),
                current_source_compared=source_root is not None,arrays_deserialized=False,pass_all=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","copy","verify"))
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--snapshot",required=True)
    parser.add_argument("--m1",action="store_true",help="Completed gradient and endpoint bridge evidence")
    parser.add_argument("--seal",help="Existing complete 36-state prediction seal, repository-relative")
    parser.add_argument("--check-source",action="store_true",help="During verify, also compare current repository bytes")
    args=parser.parse_args()
    if args.action=="verify":
        require(not args.m1 and args.seal is None,"Verify an existing snapshot without changing its target selection")
        report=verify_snapshot(args.destination,args.snapshot,source_root=ROOT if args.check_source else None)
    elif args.action=="plan":
        require(not args.check_source,"--check-source is for verify only")
        target=snapshot_path(ROOT,args.destination,args.snapshot)
        require(not target.exists(),"Snapshot ID already exists")
        report=collect_selection(ROOT,m1=args.m1,seal=args.seal)
        report.update(snapshot=args.snapshot,destination=str(target),physical_disks=physical_disks(ROOT,args.destination),
                      plan_only=True,files_written=0)
    else:
        require(not args.check_source,"--check-source is for verify only")
        report=archive_snapshot(ROOT,args.destination,args.snapshot,m1=args.m1,seal=args.seal)
    print(json.dumps({key:report[key] for key in ("snapshot","files_verified","total_bytes","pass_all","plan_only","files_written")
                      if key in report},indent=2))


if __name__=="__main__":
    main()
