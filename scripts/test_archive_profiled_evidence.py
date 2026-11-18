"""Focused synthetic byte-copy, completion, race and codec archive fixtures."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import archive_profiled_evidence as archive
from profiled_artifacts import logical_inventory, resolve_logical


def write(root,name,value):
    path=resolve_logical(root,name)
    path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(value,bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value,sort_keys=True)+"\n",encoding="utf-8")
    return path


def seed_tools(root):
    for name in ("scripts/archive_profiled_evidence.py","scripts/profiled_artifacts.py"):
        write(root,name,b"# synthetic archived tool bytes\n")


def seed_m1(root):
    seed_tools(root)
    source=write(root,"synthetic_source.py",b"# preserve CRLF bytes\r\n")
    observation=write(root,"immutable_inputs.npz",b"not an NPZ: archives must not deserialize it")
    engineering_name="manifests/lora_profiled_engineering_v2.json"
    engineering=dict(phase="engineering",artifact_sha256={"immutable_inputs.npz":archive.sha(observation)})
    engineering_path=write(root,engineering_name,engineering)
    identity=dict(input_sha256=engineering["artifact_sha256"],
                  source_sha256_lf={"synthetic_source.py":archive.sha(source,text=True)},
                  engineering_manifest_sha256=archive.sha(engineering_path))
    attempt="endpoint01"
    public=archive.PUBLIC+"/endpoint_attempts/"+attempt
    raw=archive.RAW+"/endpoint_attempts/"+attempt
    captures={}
    for label,folder in (("pristine","pristine"),("zero_lora","zero")):
        cache={}
        for name in ("forward/a/b.pth","canon_views/view_kw={'mode': 'avg-angle'}.pth"):
            path=write(root,raw+"/"+folder+"_cache/"+name,(label+name).encode())
            cache[name]=archive.sha(path)
        graph_name=public+"/"+folder+"/native_graph.json"
        graph=write(root,graph_name,dict(cache_files=cache))
        capture=dict(capture=dict(synthetic=True,variant=label),final=dict(synthetic_final=True),
                     graph_file=graph_name,graph_sha256=archive.sha(graph))
        for kind in ("native","final"):
            name=raw+"/"+folder+"/"+kind+".npz"
            path=write(root,name,(label+kind+" opaque bytes").encode())
            capture[kind+"_file"],capture[kind+"_sha256"]=name,archive.sha(path)
        write(root,public+"/"+folder+"/native.json",capture["capture"])
        write(root,public+"/"+folder+"/final.json",capture["final"])
        write(root,public+"/"+folder+"/native_trace.json",[dict(complete=True)])
        captures[label]=capture
    endpoint=dict(identity,schema="profiled_endpoint_bridge_v1",attempt=attempt,engineering_manifest=engineering_name,
        pass_all=True,original_state_unchanged=True,native_endpoint_pass=True,final_endpoint_pass=True,
        adapter_state_unchanged=True,captures=captures)
    write(root,archive.PUBLIC+"/endpoint_bridge.json",endpoint)
    write(root,public+"/comparison.json",endpoint)
    gradient_attempt=archive.PUBLIC+"/engineering/gradient02"
    gradient=dict(identity,attempt=gradient_attempt,pass_all=True,original_state_unchanged=True,
                  zero_adapter_pass=True,replay_gradient_pass=True,finite_difference_pass=True,full_macro_step_pass=True)
    write(root,archive.PUBLIC+"/gradient_bridge.json",gradient)
    write(root,gradient_attempt+"/report.json",gradient)
    write(root,gradient_attempt+"/finite_differences.json",dict(pass_all=True))
    write(root,archive.PUBLIC+"/final_calibration_checks.json",dict(pass_all=True))
    write(root,archive.PUBLIC+"/progress.json",dict(mutable=True))
    write(root,archive.RAW+"/batches/live.log",b"an unrelated live log; excluded")
    return endpoint


@contextmanager
def changed(path,content):
    before=path.read_bytes()
    path.write_bytes(content)
    try:
        yield
    finally:
        path.write_bytes(before)


def run_fixtures():
    checks=[]
    def passed(name,condition=True):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejects(name,fn):
        try:
            fn()
        except (ValueError,OSError):
            checks.append(name)
        else:
            raise AssertionError("Expected rejection: "+name)
    probe=lambda *_:dict(source=dict(drive="S",disk_number=1),destination=dict(drive="D",disk_number=2),synthetic_probe=True)
    with tempfile.TemporaryDirectory(prefix="profiled-archive-fixture-") as directory:
        base=Path(directory).resolve()
        source,destination=base/"source",base/"backup"
        source.mkdir()
        endpoint=seed_m1(source)
        plan=archive.collect_selection(source,m1=True)
        names={row["file"] for row in plan["files"]}
        passed("explicit_M1_selection_excludes_mutable_progress_and_logs", all("progress.json" not in name and not name.endswith(".log") for name in names))
        passed("opaque_NPZ_and_checkpoint_bytes_are_not_decoded", plan["arrays_deserialized"] is False)
        result=archive.archive_snapshot(source,destination,"m1",m1=True,disk_probe=probe)
        verified=archive.verify_snapshot(destination,"m1",source_root=source)
        passed("M1_copy_and_both_disk_byte_verification",result["pass_all"] and verified["files_verified"]==len(names))
        passed("verification_returns_requested_snapshot_identity",verified["snapshot"]==result["snapshot"]=="m1")
        data=destination/"snapshots/m1/data"
        logical=logical_inventory(data)
        passed("logical_colon_cache_name_and_CRLF_bytes_preserved", any(":" in name for name in logical) and
               resolve_logical(data,"synthetic_source.py").read_bytes()==b"# preserve CRLF bytes\r\n")
        rejects("existing_snapshot_never_overwritten",lambda:archive.archive_snapshot(source,destination,"m1",m1=True,disk_probe=probe))
        completion=destination/"snapshots/m1/manifest.json"
        changed_report=dict(archive.read(completion),snapshot="wrong-identity")
        with changed(completion,json.dumps(changed_report).encode()):
            rejects("completion_manifest_must_match_original_copy_intent",lambda:archive.verify_snapshot(destination,"m1"))
        rejects("archive_root_inside_source_rejected",lambda:archive.archive_snapshot(source,source/"nested","unsafe",m1=True,disk_probe=probe))
        rejects("archive_root_containing_source_rejected",lambda:archive.archive_snapshot(source,base,"unsafe",m1=True,disk_probe=probe))
        original_name=endpoint["captures"]["pristine"]["native_file"]
        backed=resolve_logical(data,original_name)
        with changed(backed,b"corrupted backup"):
            rejects("tampered_destination_rejected",lambda:archive.verify_snapshot(destination,"m1"))
        marker=resolve_logical(source,archive.PUBLIC+"/endpoint_bridge.json")
        bad=dict(endpoint,pass_all=False)
        with changed(marker,json.dumps(bad).encode()):
            rejects("failed_M1_marker_prevents_copy",lambda:archive.archive_snapshot(source,destination,"failed-proof",m1=True,disk_probe=probe))
        passed("bad_preflight_creates_no_snapshot",not (destination/"snapshots/failed-proof").exists())
        target=resolve_logical(source,original_name)
        before=target.read_bytes()
        def mutate_source(count,row):
            if count==1:
                target.write_bytes(b"changed while the archive was copying")
        try:
            rejects("source_mutation_during_copy_rejected",lambda:archive.archive_snapshot(source,destination,"mutated-source",
                    m1=True,disk_probe=probe,after_copy=mutate_source))
        finally:
            target.write_bytes(before)
        failed=destination/"snapshots/mutated-source"
        passed("failed_copy_preserves_partial_bytes_without_completion_manifest",(failed/"failure.json").is_file()
               and (failed/"data").is_dir() and not (failed/"manifest.json").exists())
        rejects("failed_snapshot_is_not_verified_as_complete",lambda:archive.verify_snapshot(destination,"mutated-source"))
        def mutate_destination(count,row):
            if count==1:
                resolve_logical(destination/"snapshots/mutated-destination/data",row["file"]).write_bytes(b"bad backup")
        rejects("destination_mutation_before_final_manifest_rejected",lambda:archive.archive_snapshot(source,destination,
                "mutated-destination",m1=True,disk_probe=probe,after_copy=mutate_destination))
        passed("archive_utility_performs_no_deletions",result["files_deleted"]==0 and failed.exists())

        # Reuse only the synthetic 36-case constructor, never the study scorer.
        from test_profiled_scoring import fixture_tree, write as fixture_write
        from seal_profiled_predictions import build_seal
        scientific=base/"scientific"
        scientific.mkdir()
        registry,evaluation=fixture_tree(scientific)
        seed_tools(scientific)
        seal=build_seal(scientific,"registry.json",evaluation)
        fixture_write(scientific/"seal.json",seal)
        m2=archive.collect_selection(scientific,seal="seal.json")
        passed("M2_requires_and_preserves_all_36_sealed_states",m2["targets"][0]["completed_states"]==36)
        result2=archive.archive_snapshot(scientific,base/"backup-m2","sealed36",seal="seal.json",disk_probe=probe)
        passed("M2_36_state_archive_verifies_without_arrays",archive.verify_snapshot(base/"backup-m2","sealed36")["pass_all"]
               and result2["arrays_deserialized"] is False)
        partial=dict(seal,completed_states=35)
        with changed(scientific/"seal.json",json.dumps(partial).encode()):
            rejects("partial_M2_seal_rejected",lambda:archive.collect_selection(scientific,seal="seal.json"))
        lock=write(scientific,archive.RAW+"/scientific_gpu.lock",dict(run_id=registry["jobs"][0]["run_id"]))
        rejects("actively_owned_case_rejected_even_with_completion_record",lambda:archive.collect_selection(scientific,seal="seal.json"))
        write(source,archive.RAW+"/scientific_gpu.lock",dict(run_id="unrelated_current_case"))
        passed("unrelated_M2_activity_does_not_block_immutable_M1_selection",archive.collect_selection(source,m1=True)["targets"][0]["kind"]=="M1")

    # Production disk guard tested independently with no actual disk or shell IO.
    source_drive=SimpleNamespace(resolve=lambda:SimpleNamespace(drive="C:"))
    target_drive=SimpleNamespace(resolve=lambda:SimpleNamespace(drive="E:"))
    with patch.object(archive,"os",SimpleNamespace(name="nt")):
        with patch.object(archive.subprocess,"check_output",side_effect=["1","1"]):
            rejects("same_physical_disk_rejected_despite_different_letters",lambda:archive.physical_disks(source_drive,target_drive))
        with patch.object(archive.subprocess,"check_output",side_effect=["1","2"]):
            passed("distinct_physical_disks_verified",archive.physical_disks(source_drive,target_drive)["destination"]["disk_number"]==2)
    return dict(pass_all=True,synthetic_only=True,checks=checks,checks_passed=len(checks),
                real_evidence_copied=False,real_arrays_decoded=False,gpu_used=False)


if __name__=="__main__":
    print(json.dumps(run_fixtures(),indent=2))
