"""CPU tests for the early-stage schedule, physical parameter identity and exact Adam aging."""
from contextlib import nullcontext
from unittest.mock import patch
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None,"CPU PyTorch not installed")
class ScheduleTest(unittest.TestCase):
    def exercise(self, arm, refresh, prefix_tensors):
        from tco_two_track_hybrid import HybridAdamAudit, stage_number, move_frozen_base
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(torch.nn.Linear(3,3) for _ in range(4))
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.aggregator = torch.nn.Module()
                agg = self.aggregator
                agg.aa_block_num,agg.aa_block_size,agg.aa_order = 24,1,["frame","global"]
                agg.frame_blocks = torch.nn.ModuleList(Block() for _ in range(24))
                agg.global_blocks = torch.nn.ModuleList(Block() for _ in range(24))
                self.depth_head = torch.nn.Module()
                self.depth_head.intermediate_layer_idx = (4,11,17,23)
            def get_lora_target_modules(self):
                return [*self.aggregator.frame_blocks,*self.aggregator.global_blocks]
            def extract_patch_tokens(self,_):
                raise AssertionError("No image path needed")
            def forward_original_vggt(self,_):
                raise AssertionError("No camera path needed")
            def backbone_peft_setup(self):
                params = []
                for block in self.get_lora_target_modules():
                    for layer in block.layers:
                        layer.lora_A = torch.nn.Parameter(torch.ones(1,3))
                        layer.lora_B = torch.nn.Parameter(torch.zeros(3,1))
                        params.extend([layer.lora_A,layer.lora_B])
                return torch.optim.Adam(params,lr=.001),params
        model = Net()
        audit = HybridAdamAudit(model,arm,42,"real")
        with patch.object(torch.random,"fork_rng",return_value=nullcontext()), patch.object(torch.cuda,"current_device",return_value=0):
            adam,_ = model.backbone_peft_setup()
        for step in range(1,11):
            if step in refresh:
                self.assertFalse(audit.path.cache)
            self.assertEqual(audit.path.start,(11 if arm.startswith("Narrow") else 0) if step in refresh else 12)
            self.assertEqual(audit.path.reuse,step not in refresh and not arm.endswith("Uncached"))
            for name,p in audit.factors.items():
                if p.requires_grad:
                    p.grad = torch.ones_like(p)
            if step not in refresh:
                audit.path.cache["simulated"] = True
            adam.step()
            move_frozen_base(model.aggregator.global_blocks[11],"cpu")
        self.assertEqual(audit.finish(),{"prefix_"+str(len(refresh)):prefix_tensors,"suffix_10":192})
        self.assertEqual(audit.path.generation,len(refresh))
        self.assertEqual({id(p) for n,p in model.named_parameters() if n.endswith((".lora_A",".lora_B"))},
                         {id(p) for p in adam.param_groups[0]["params"]})

    def test_narrow_caches_only_immutable_stages(self):
        from tco_two_track_hybrid import NarrowPath, RefreshPath
        cached = object.__new__(NarrowPath)
        cached.stable, cached.stable_key, cached.arm = {}, ("cpu",False), "Narrow"
        cached.stable_hits = cached.stable_builds = 0
        visits = []
        upstream_delta = [0.]
        def execute(_,i,tokens,pos,taps,b,v,c):
            visits.append(i)
            z = tokens + (upstream_delta[0] if i==11 else .1)
            if i in (4,11):
                taps[i] = z
            return z
        def invoke():
            z,taps = torch.ones(2),[None]*24
            for i in range(12):
                z = cached.stage(i,z,None,taps,1,1,False)
            return z,taps[4],taps[11]
        with patch.object(RefreshPath,"stage",execute):
            before = invoke()
            self.assertEqual(visits,list(range(12)))
            visits.clear()
            upstream_delta[0] = 1.
            after = invoke()
            self.assertEqual(visits,[11])
            self.assertTrue(torch.equal(before[1],after[1]))
            self.assertTrue(torch.equal(before[2]+1,after[2]))
            cached.arm = "NarrowUncached"
            fresh = invoke()
            for reference,test in zip(fresh,after):
                self.assertTrue(torch.equal(reference,test))
        self.assertEqual((cached.stable_builds,cached.stable_hits),(1,1))

    def test_refresh_once(self):
        self.exercise("Refresh1",{1},192)

    def test_refresh_again(self):
        self.exercise("Refresh2",{1,6},192)

    def test_narrow_uncached(self):
        self.exercise("NarrowUncached",{1},16)


if __name__ == "__main__":
    unittest.main()
