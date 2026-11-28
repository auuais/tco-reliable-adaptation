"""Generated receipts only. Demonstrate the evidential boundary for local distillation closure."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import close_tco_two_track_distillation as dep


def history(kind,seconds=1201):
    return dict(status=kind,stopped=True,rental_compute_usd=0,wall_seconds=seconds,
                error="command timed out" if kind=="timeout" else None,verified_cuda_oom=kind=="oom")


class Closure(unittest.TestCase):
    def test_timeout_needs_a_complete_unchanged_retry_and_real_deadline(self):
        with self.assertRaises(ValueError):
            dep.final_state([history("timeout")],1200)
        with self.assertRaises(ValueError):
            dep.final_state([history("timeout"),history("timeout",13)],1200)
        self.assertEqual(dep.final_state([history("timeout"),history("timeout")],1200),"timeout")
        self.assertEqual(dep.final_state([history("timeout"),history("completed")],1200),"completed")

    def test_missing_or_recoverable_program_error_is_not_capacity_evidence(self):
        for rows in ([],[history("error")],[history("oom")|{"verified_cuda_oom":False}],
                     [history("provisioning"),history("provisioning")],[history("oom"),history("completed")]):
            with self.assertRaises(ValueError):
                dep.final_state(rows,1200)
        self.assertEqual(dep.final_state([history("oom")],1200),"oom")

    def test_verified_teacher_failure_preserves_qualifying_hybrid(self):
        with TemporaryDirectory() as temp,ExitStack() as patches,redirect_stdout(io.StringIO()):
            folder=Path(temp)
            cfg=folder/"master.json"
            trigger=folder/"trigger.json"
            entry=folder/"a_hybrid_stress_entry.json"
            dep.seal(cfg,dict(protocol="fixture"))
            dep.seal(entry,dict(candidate="Refresh1"))
            dep.seal(trigger,dict(proceed=True,hybrid="Refresh1",stress_pass=True,development_sha256=dep.sha(entry)))
            sample=[dict(id=f"teacher{i}",arm="Full40",status="oom" if i==2 else "completed") for i in range(6)]
            for k,v in (("PUBLIC",folder),("MANIFEST",cfg),("TRIGGER",trigger)):
                patches.enter_context(patch.object(dep,k,v))
            for k,v in (("intake_proof",{"six":"fixture"}),("model_rows",sample),
                 ("hybrids",(["development"],["Refresh1"],{"comparison":"fixture"})),
                 ("choose_a","Refresh1"),("analyze",{"stress":"fixture"})):
                patches.enter_context(patch.object(dep,k,return_value=v))
            dep.close("teachers")
            evidence=dep.read(folder/"a_distillation_local_inconclusive.json")
            nominee=dep.read(folder/"a_nomination.json")
            self.assertFalse(evidence["offline_student_rows"])
            self.assertEqual(evidence["decisive_slots"],["teacher2"])
            self.assertEqual(nominee["candidate"],"Refresh1")
            self.assertEqual(dep.sha(folder/"a_distillation_local_inconclusive.json"),nominee["distillation_resource_closure_sha256"])
            dep.close("teachers")

    def test_successful_teachers_cannot_be_closed_as_missing(self):
        with TemporaryDirectory() as tmp,patch.object(dep,"TRIGGER",Path(tmp)/"gate.json"),patch.object(dep,"intake_proof",return_value={}),patch.object(dep,"model_rows",return_value=[dict(status="completed")]*6):
            dep.seal(dep.TRIGGER,dict(proceed=True))
            with self.assertRaisesRegex(ValueError,"All targeted"):
                dep.close("teachers")


if __name__=="__main__":
    unittest.main()
