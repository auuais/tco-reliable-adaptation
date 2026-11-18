"""Write the paired effects, fixed controls and coverage for the step-selection study."""
from collections import Counter

from replay_native_lora_step import PUBLIC, ROOT, read


def number(value, signed=False):
    return "—" if value is None else f"{value:+.3f}" if signed else f"{value:.3f}"


def report():
    seal = read(PUBLIC / "selection_seal.json")
    result = read(PUBLIC / "held_scores.json")
    audit = read(PUBLIC / "independent_replay.json")
    manifest = read(ROOT / "manifests/native_lora_step_v1.json")
    cases = list(result["cases"].values())
    choices = Counter(c["chosen"] for c in cases)
    delta = [c["delta_mm"] for c in cases if c["delta_mm"] is not None]
    wins, losses = sum(v < -.05 for v in delta), sum(v > .05 for v in delta)
    regrets = [c["errors_mm"]["selected"] - min(c["errors_mm"][key] for key in manifest["states"])
               for c in cases if c["valid"]]
    headings = {
        "consistent_gain": "Adaptation-based step selection beat the fixed full step in the registered room screen.",
        "mixed_or_small_gain": "The result missed the registered threshold for a consistent gain over the fixed full step.",
        "indeterminate": "The full cohort could not be evaluated under the registered validity rule.",
    }
    lines = [
        "# MASt3R LoRA step selection on fresh rooms", "", headings[result["verdict"]],
        f"Selection improved held depth error by more than 0.05 mm in {wins} cases and worsened it by more than "
        f"0.05 mm in {losses}. The remaining {len(delta)-wins-losses} valid paired changes fell within that range.",
        "", "## Paired held depth errors",
        "",
        "Every value is mean absolute axial depth error in millimeters after fresh native alignment and a shared "
        "metric gauge convention. The final column subtracts the fixed full step from the selected result.",
        "",
        "| Room | Adaptation source | Chosen multiplier | Fixed 0 | Fixed 0.25 | Fixed 0.5 | Fixed 1 | Selected | Change |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        values = [number(case["errors_mm"][model]) for model in ("zero", "quarter", "half", "full", "selected")]
        step = manifest["states"][case["chosen"]]
        row = [case["room"], case["side"], f"{step:g}", *values, number(case["delta_mm"], True)]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "", "## Equal-room means",
        "", "| Adaptation source | Fixed 0 | Fixed 0.25 | Fixed 0.5 | Fixed 1 | Selected | Gain vs fixed 1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for side, row in result["directional"].items():
        errors = [number(row["errors_mm"][state]) for state in ("zero", "quarter", "half", "full", "selected")]
        gain = row["relative_gain_vs_full"]
        lines.append("| " + " | ".join([side, *errors, "—" if gain is None else f"{gain:+.3%}"]) + " |")
    held = [v["count"] for case in cases for v in case["held_coverage"]]
    selector = [n for env in manifest["inputs"]["environments"].values() for side in env["packets"].values()
                for n in side["counts"][2]]
    cases_sealed = list(seal["cases"].values())
    seconds = sum(c["launch"]["wall_seconds"] + sum(r["wall_seconds"] for r in c["retired_attempts"])
                  for c in cases_sealed)
    peak = max(max(c["direction"]["peak_allocated_mib"], *(r["peak_allocated_mib"] for r in c["states"].values()))
               for c in cases_sealed)
    counts = ", ".join(f"{choices[state]} at {manifest['states'][state]:g}" for state in manifest["states"])
    both = ", ".join(manifest["rooms"][env] for env in result["both_directions_improved"]) or "none"
    lines += [
        "", "## Coverage and verification", "",
        f"- All {len(held)} fixed held views retained {min(held)}–{max(held)} unique pixels after overlap removal. "
        f"The reserved adaptation selector used {min(selector)}–{max(selector)} observations per view.",
        f"- The distribution of selected multipliers was {counts}. Rooms improving in both directions by "
        f"more than 0.05 mm were {both}.",
        "- The registered consistency criterion required at least 1% mean gain in both directions, two rooms "
        "with gains in both directions, and no case with a regression greater than 1%.",
        f"- The independent replay verified {audit['held_view_rows']} public held-view rows and "
        f"{audit['verified_local_view_scores']} local held and selector view scores, including the metric gauge "
        "and fixed overlap masks.",
        f"- All {len(cases_sealed)*len(manifest['states'])} native captures finished on the local RTX 3080. "
        f"GPU jobs used {seconds/60:.1f} summed minutes with {peak/1024:.2f} GiB peak allocated memory. RunPod compute was USD 0.",
        "- The three rooms each contribute two paired adaptation directions over the same RGB views. "
        "The results measure first-step sparse depth accuracy.",
        "",
        "[Prospective design](../../docs/NATIVE_LORA_STEP_SELECTION.md) · "
        "[Execution manifest](../../manifests/native_lora_step_v1.json) · [Pre-score seal](selection_seal.json) · "
        "[Scores](held_scores.json) · [Replay receipt](independent_replay.json)",
        "", "## Reproduce", "",
        "```powershell",
        "python scripts/native_lora_step_inputs.py intake",
        "python scripts/run_native_lora_step_local.py",
        "python scripts/score_native_lora_step.py seal",
        "python scripts/score_native_lora_step.py score",
        "python scripts/replay_native_lora_step.py --private",
        "```",
        "",
        "Run the commands in a fresh working copy of preregistration commit `52a64f1` with the three publisher "
        "ZIP archives. The launch uses the registered checkpoint and Docker image. "
        "To replay the aggregate decision from Git alone, run "
        "`python scripts/replay_native_lora_step.py`.",
        "",
    ]
    if regrets:
        offset = lines.index("## Coverage and verification")
        closest = sum(gap <= 1e-10 for gap in regrets)
        lines[offset:offset] = [
            f"The selected multiplier matched the best of the four tested held-out endpoints in "
            f"{closest} of {len(regrets)} valid cases. Its largest gap to that retrospective benchmark "
            f"was {max(regrets):.3f} mm.", "",
        ]
    (PUBLIC / "README.md").write_bytes("\n".join(lines).encode("utf-8"))


if __name__ == "__main__":
    report()
