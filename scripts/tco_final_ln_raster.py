"""Use exactly one pinned gsplat RGB+ED dispatch for each ED/RGB pair."""
KEYS = ("means", "quats", "scales", "opacities", "viewmats", "Ks")


def signature(t):
    return (t.data_ptr(), t.storage_offset(), tuple(t.shape), t.stride(), t.dtype, t.device)


class RasterPair:
    def __init__(self, actual, render, record):
        self.actual, self.render, self.record = actual, render, record
        self.pending, self.rgb = None, None
        self.calls, self.pairs = [], 0

    def launch(self, kw, physical):
        self.calls.append(physical)
        self.record("raster_enter", mode=physical, pair=self.pairs + 1)
        out = self.actual(**dict(kw, render_mode=physical))
        self.record("raster_return", mode=physical, pair=self.pairs + 1)
        return out

    def __call__(self, **kw):
        mode, colors = kw["render_mode"], kw["colors"]
        geometry = tuple(signature(kw[k]) for k in KEYS)
        settings = {k: v for k, v in kw.items() if k not in (*KEYS, "colors", "render_mode")}
        if mode == "ED":
            if self.pending is not None or not all(kw[k].requires_grad for k in ("means", "opacities")):
                raise ValueError("Unexpected ED order or detached geometry")
            self.pending = (geometry, settings, signature(colors))
            if self.render == "separate":
                return self.launch(kw, "ED")
            if colors.shape[-1] != 3 or kw["packed"] or "backgrounds" in kw:
                raise ValueError("Unexpected RGB+ED colors or settings")
            out = self.launch(kw, "RGB+ED")
            if len(out) != 7 or out[0].shape[-1] != 4:
                raise ValueError("Unexpected RGB+ED output")
            self.rgb = out[0][..., :3]
            return (out[0][..., 3:], *out[1:])
        if mode != "RGB" or self.pending is None or self.pending[:2] != (geometry, settings):
            raise ValueError("Native ED/RGB geometry, camera or rendering settings differ")
        if self.render == "separate":
            out = self.launch(kw, "RGB")
        else:
            if self.pending[2] != signature(colors) or self.rgb is None:
                raise ValueError("Native RGB colors differ from combined dispatch")
            out = (self.rgb, None, None, None, None, None, None)
        self.rgb, self.pending = None, None
        self.pairs += 1
        return out

    def check(self, n):
        pattern = ["ED", "RGB"] if self.render == "separate" else ["RGB+ED"]
        if self.rgb is not None or self.pending is not None or self.pairs != n or self.calls != pattern * n:
            raise ValueError(f"Unexpected physical dispatch order at update {n}: {self.calls}")


def cpu_vjp():
    """Check four cycles with signed gradients and a detached-alpha sensitivity control."""
    import torch
    def replay(mode, detach=False):
        def actual(**kw):
            means, opacity, colors = kw["means"], kw["opacities"], kw["colors"]
            alpha = opacity.sigmoid().reshape(1, -1, 1)
            z = means[:, 2].reshape(1, -1, 1)
            rgb = colors * (opacity + means[:, 0]).reshape(1, -1, 1)
            vals = {"RGB": rgb, "ED": z, "RGB+ED": torch.cat((rgb, z), dim=-1)}
            return vals[kw["render_mode"]], alpha.detach() if detach else alpha, None, None, None, None, None
        pair = RasterPair(actual, mode, lambda *a, **kw: None)
        outputs = []
        for n in range(4):
            x = (torch.arange(12, dtype=torch.float64).reshape(4, 3)/9 + n).requires_grad_()
            o = torch.linspace(-1., 1., 4, dtype=torch.float64, requires_grad=True)
            c = (x.detach().clone() / 7).unsqueeze(0)
            kw = dict(means=x, quats=x, scales=x, opacities=o, viewmats=x, Ks=x, packed=False)
            ed = pair(**kw, colors=c if mode == "joint" else c[..., :1], render_mode="ED")
            rgb = pair(**kw, colors=c, render_mode="RGB")
            loss = ((rgb[0] - 0.2).abs().mean(-1, keepdim=True) * ed[1] * (ed[0] > -3)).sum() / ed[1].sum()
            outputs.append((loss.detach(), *torch.autograd.grad(loss, (x, o))))
            pair.check(n + 1)
        return outputs
    native, merged, negative = replay("separate"), replay("joint"), replay("joint", True)
    if any(not torch.equal(a, b) for refs, opts in zip(native, merged) for a, b in zip(refs, opts)):
        raise ValueError("CPU forward or signed VJP differs")
    if any(torch.equal(ref[2], neg[2]) for ref, neg in zip(native, negative)):
        raise ValueError("Detached-alpha sensitivity check failed")
    return dict(cycles=4, exact_loss=True, exact_signed_means_and_opacity=True, detached_alpha_detected=True)
