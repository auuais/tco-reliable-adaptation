"""Small synthetic decisions. No actual room, teacher, model, or scorer inputs are opened."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from unittest.mock import patch

import lock_tco_two_track_nominees as mod


def a_decision(passed):
    entry=dict(equal_step={"passed":passed},comparisons={"Full10":{"metrics":{
        "scene_seconds":{"ratio":.8},"peak_allocated_bytes":{"ratio":.9}}}})
    analysis={"candidates":{"Refresh1":entry}}
    return dict(registration_sha256="r",candidate="Refresh1" if passed else None,development=analysis,
                stress=analysis if passed else None,stages=[],stress_stages=[])


def time_lock(ordinary,candidate):
    rows=[]
    times={}
    for r in mod.DEV:
        for s in mod.SEEDS:
            for a,t in [("Refresh1",candidate)]+[(f"Full{k}",float(k)) for k in ordinary]:
                times[a,r,s]=t
                rows.append(dict(id=f"{a}/{r}/{s}",arm=a,room=r,seed=s,scene_seconds=t,model_receipt_sha256="a"*64))
    k,ratios=mod.select_count(times,mod.DEV,mod.SEEDS,"Refresh1",ordinary)
    return dict(candidate="Refresh1",nomination_sha256="nom",comparator_policy_sha256="policy",
                comparator_k=k or min(ordinary),selected_k=k,affordable=k is not None,timing_receipts=rows,
                ratios={str(i):v for i,v in ratios.items()})


class Nominees(unittest.TestCase):
    def test_positive_needs_confirmed_matched_timing_and_rejects_relabel(self):
        t=time_lock([4,5,6,10],5.)
        self.assertEqual(mod.verify_time(t,"Refresh1","nom","policy"),dict(arm="Full5",steps=5,affordable=True))
        with self.assertRaisesRegex(ValueError,"Inconsistent"):
            mod.verify_time(dict(t,comparator_k=6),"Refresh1","nom","policy")
        with self.assertRaisesRegex(ValueError,"Wrong nominee"):
            mod.verify_time(t,"Refresh1","changed","policy")
        t=time_lock([1,2,3,4,5,6,10],.2)
        self.assertFalse(mod.verify_time(t,"Refresh1","nom","policy")["affordable"])

    def test_pilot_does_not_freeze_a_passing_b_or_a_without_stress(self):
        a=a_decision(True)
        b=dict(registration_sha256="r",gates={"coverage_complete":True,"accuracy":False},passed=False)
        i=dict(coverage={"expected":24,"scored":24,"failed":[],"missing":[]},eligible=False)
        self.assertEqual(mod.select(a,b,i,"r"),"Refresh1")
        with self.assertRaisesRegex(ValueError,"missing its stress"):
            mod.select(dict(a,stress=None),b,i,"r")
        with self.assertRaisesRegex(ValueError,"passing B pilot"):
            mod.select(a,dict(b,gates={"coverage_complete":True},passed=True),dict(i,eligible=True),"r")

    def test_negative_publishes_replayable_note_without_timing_or_enrollment(self):
        with TemporaryDirectory() as tmp, ExitStack() as patches:
            root=Path(tmp)
            pub=root/"results"
            master=root/"master.json"
            policy=root/"policy.json"
            rooms=("x","y","z")
            mod.seal(master,dict(study=mod.STUDY,
                b={"dev_triplets":[dict(scene=r,anchor=42,seeds=[42,43,44]) for r in rooms]},
                fresh={"primary":{"source_revision":"a"*40,"visits":6,"seeds":mod.NEW_SEEDS}},
                runtime={"checkpoint":"c"*64,"upstream":"b"*40,"image":"sha256:"+"1"*64}))
            mod.seal(policy,dict(test=True))
            reg=mod.digest(master)
            mod.seal(pub/"a_nomination.json",dict(a_decision(False),registration_sha256=reg))
            cases=[dict(room=r,arm=a,seed=s,accuracy_mm=2.,completeness_mm=3.) for r in rooms
                   for a,seeds in (("native",[42,43,44]),("reprojected",[42,43,44]),("fusion",[None]),
                                   ("triplicate_anchor",[42])) for s in seeds]
            provenance=dict(study=mod.STUDY,registration_sha256=reg,source_lock_sha256="f"*64)
            mod.seal(pub/"b_development_rows.json",dict(**provenance,rows=cases))
            mod.seal(pub/"b_development_decision.json",dict(**provenance,gates={"coverage_complete":True,"accuracy":False},
                      rows_sha256=mod.digest(pub/"b_development_rows.json"),passed=False))
            mod.seal(pub/"b_development_independent.json",dict(study=mod.STUDY,
                coverage={"expected":24,"scored":24,"failed":[],"missing":[]},eligible=False))
            for k,v in (("ROOT",root),("PUBLIC",pub),("MANIFEST",master),("POLICY",policy),("CODE",())):
                patches.enter_context(patch.object(mod,k,v))
            commit=patches.enter_context(patch.object(mod,"committed",return_value={"upstream_code_files":0}))
            mod.write()
            mod.write()
            lock=mod.read(pub/"nominees.json")
            self.assertFalse(lock["enrollment_authorized"])
            self.assertIsNone(lock["comparable_ordinary"])
            self.assertTrue((pub/"no_fresh_evaluation.md").exists())
            self.assertEqual(commit.call_count,2)
            self.assertFalse(any("arkit" in p.name for p in pub.iterdir()))

    def test_separate_git_owners_and_pinned_compatibility(self):
        def git(owner,*args):
            return subprocess.check_output(["git","-c","user.name=fixture","-c","user.email=fixture@example.invalid",
                   "-c","core.autocrlf=false","-c","commit.gpgsign=false",*args],cwd=owner,stderr=subprocess.PIPE,text=True).strip()

        with TemporaryDirectory() as tmp,ExitStack() as patches:
            root=Path(tmp).resolve()
            vendor=root/"third_party/TCO"
            vendor.mkdir(parents=True)
            pub=root/"public"
            master=root/"manifests/registration.json"
            old=root/"manifests/tco_one_percent_v1.json"
            for repo in (root,vendor):
                git(repo,"init","-q")
            (root/".gitignore").write_bytes(b"third_party/\n")
            pristine=vendor/"native.py"
            compat=vendor/"compat.py"
            for path in (pristine,compat):
                path.write_bytes(b"value = 1\n")
            git(vendor,"add",".")
            git(vendor,"commit","-qm","upstream")
            rev=git(vendor,"rev-parse","HEAD")
            compat.write_bytes(b"value = 2\n")
            mod.seal(master,{"runtime":{"upstream":rev}})
            mod.seal(old,{"upstream":rev,"source_sha256_lf":{p.relative_to(root).as_posix():mod.lf_sha(p)
                                                            for p in (pristine,compat)}})
            mod.seal(pub/"runtime.json",{"upstream_commit":rev,"upstream_modified_files":{"compat.py":mod.digest(compat)}})
            git(root,"add",".")
            git(root,"commit","-qm","scientific pins")
            for k,v in (("ROOT",root),("PUBLIC",pub),("MANIFEST",master)):
                patches.enter_context(patch.object(mod,k,v))
            self.assertEqual(mod.committed([pristine,compat])["upstream_code_files"],2)
            with self.assertRaises(subprocess.CalledProcessError):
                mod.committed([vendor/"unregistered.py"])
            pristine.write_bytes(b"value = 0\n")
            with self.assertRaises(subprocess.CalledProcessError):
                mod.committed([pristine,compat])
            git(vendor,"restore","--","native.py")
            compat.write_bytes(b"value = 3\n")
            with self.assertRaisesRegex(ValueError,"compatibility patch changed"):
                mod.committed([pristine,compat])
            git(vendor,"commit","--allow-empty","-qm","wrong revision")
            with self.assertRaisesRegex(ValueError,"upstream checkout"):
                mod.committed([pristine,compat])


if __name__=="__main__":
    unittest.main()
