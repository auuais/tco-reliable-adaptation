"""Render the study report from its sealed measurements."""
from statistics import mean

from run_depth_scale_study import PUBLIC
from score_depth_scale_study import HISTORICAL, LOCAL, METRICS, REFS, read


LABEL = dict(frozen='Frozen VGGT, FP32', direct='Selected focal, unchanged depth', shared='One shared depth scale',
             five='Five depth scales', guarded='Five scales with validation', native_3090='Archived frozen VGGT, native',
             tco_retain='Archived TCO, retain', tco_consensus='Archived TCO, consensus', tco_adaptive='Archived TCO, adaptive')


def main():
    rows = read(PUBLIC/'predictions.json')['rows']
    scores = read(PUBLIC/'scores.json')
    summary = scores['results']
    main = summary['references'][REFS[0]]
    released = summary['references'][REFS[1]]['overall']
    best,historical = summary['local_comparator'],summary['historical_comparator']
    accepted = sum(r['decision']['decision']=='accepted' for r in rows)
    archive = read(PUBLIC/'archive.json')
    outcome = 'passed' if summary['advance'] else 'failed'
    def effect(gain):
        return f'{abs(gain):.2%} '+('lower' if gain>=0 else 'higher')
    lines = ['# Can five depth scales improve frozen VGGT reconstruction?', '',
        f'The guarded method produced **{effect(summary["local_gain"])} registered 3D error** than the strongest local control, '
        f'{LABEL[best].lower()}. Its error was **{effect(summary["historical_gain"])}** than the strongest historical control '
        f'({LABEL[historical]}). It {outcome} the preregistered development gate. '
        '[Registered protocol](../../docs/DEPTH_SCALE_PROTOCOL.md) · [Full measurements](scores.json)', '',
        '## Geometry on the two depth references','',
        'Lower scores and distances are better. The score averages accuracy and completeness relative to frozen local FP32 '
        'for each condition and then averages the twelve conditions equally. The distances and normal consistency below '
        'use the RGB-registered depth reference. Higher normal consistency is better.', '',
        '| Method | Registered score | Accuracy (mm) | Completeness (mm) | Normal consistency | Released score |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in LOCAL+HISTORICAL:
        d = main['overall'][arm]
        acc,comp,norm = (d['raw'][k] for k in METRICS)
        lines.append(f'| {LABEL[arm]} | {d["score"]:.4f} | {1000*acc:.2f} | {1000*comp:.2f} | {norm:.4f} | {released[arm]["score"]:.4f} |')
    lines += ['', 'The historical frozen arm uses the native mixed-precision RTX 3090 coordinates. Each historical TCO arm '
        'averages two independent 40-update trajectories per condition. The local arms share frozen RTX 3080 depth and pose '
        'heads. Their corrected coordinates are built in float64 and saved in float32. The frozen control uses the original '
        'float32 unprojection. Hardware and arithmetic differences matter for interpreting cross-execution comparisons. '
        '[Matched arithmetic control](../matched_focal_heads/RESULTS.md)', '', '## RGB validation and dense accuracy', '',
        f'The held-back RGB matches accepted {accepted} of 12 proposals. Each proposal changes at most five scalar values '
        'and keeps the frozen poses and within-view depth shape. A veto returns the selected focal with identity depth scales. '
        'The validation fold contains separate matched features from the same five images. '
        'The three development sequences were examined in earlier experiments.', '',
        '| Scene | Condition | Focal direction | Held-back loss reduction | Decision | Guarded registered score |',
        '|---|---|---|---:|---|---:|']
    for p,r in zip(rows,scores['rows']):
        d = p['decision']
        reduction = f'{1-d["after"]/d["before"]:.2%}' if d['before'] else 'unavailable'
        c = r['metrics'][REFS[0]]
        normalized = mean(c['guarded'][m]/c['frozen'][m] for m in METRICS[:2])
        case = 'Metadata fault' if r['case']=='metadata_fault' else 'Valid zoom'
        direction = 'Shorter' if r['direction']<0 else 'Longer'
        lines.append(f'| {r["scene"]} | {case} | {direction} | {reduction} | {d["decision"].replace("_"," ")} | {normalized:.4f} |')
    worst = min(summary['gain_by_scene'],key=summary['gain_by_scene'].get)
    lines += ['',f'The weakest scene relative to the strongest aggregate local control was {worst}, with '
        f'{effect(summary["gain_by_scene"][worst])} error. The two metadata directions reuse the same unmodified images. '
        'The experiment contains nine unique five-view RGB batches across three scenes. Its twelve conditions do not '
        'supply twelve independent scene replicates.', '', '## Corresponding-pixel diagnostic','']
    diagnostic = read(PUBLIC/'pixel_diagnostic.json')['summary']
    for reference,display in zip(REFS,('RGB-registered depth','Released depth')):
        vals = diagnostic[reference]['overall']
        lines.append(f'{display} gave a condition-normalized corresponding-pixel mean 3D distance of '
                     f'{vals["native"]["relative_mean"]:.4f} for native frozen coordinates, 1.0000 for frozen FP32 '
                     f'and {vals["guarded"]["relative_mean"]:.4f} for the guarded result. The respective raw mean '
                     f'distances were {1000*vals["native"]["mean_m"]:.2f}, {1000*vals["fp32"]["mean_m"]:.2f} '
                     f'and {1000*vals["guarded"]["mean_m"]:.2f} mm.')
        lines.append('')
    lines += ['This secondary diagnostic aligns the point maps by one global similarity in double precision and '
        'compares predictions with depth at the same pixels before ICP. It supplies another measure of reconstruction '
        'quality on these registered inputs. The spending decision uses the originally registered surface metrics. '
        '[Paired-pixel measurements and RMSE](pixel_diagnostic.json)', '', '## How much can rigid camera corrections recover?', '',
        'Using the reference coordinates for this diagnostic, we fitted increasingly flexible transforms to exactly '
        'the same corresponding pixels. The values are the mean of condition-level optimal RMSEs in millimeters '
        'on the direct-focal predictions. Each added degree of freedom can only lower the fitted squared error.', '',
        '| Reference | One similarity | Independent translations | Independent poses, shared scale | Independent similarities | Independent affine transforms |',
        '|---|---:|---:|---:|---:|---:|']
    capacity = read(PUBLIC/'geometry_headroom.json')['results']['overall_rmse_m']
    for ref,title in zip(REFS,('Registered RGB','Released')):
        vals = capacity[ref]['direct']
        lines.append('| '+title+' | '+' | '.join(f'{1000*vals[k]:.2f}' for k in ('global','translations','rotations','scales','affine'))+' |')
    reg = capacity[REFS[0]]['direct']
    lines += ['',f'Even with reference geometry available, independently aligning and scaling each view reduces the '
        f'average registered pixel RMSE by only {1-reg["scales"]/reg["global"]:.2%}. Allowing independent scales '
        f'after the rotations and translations accounts for {1000*(reg["rotations"]-reg["scales"]):.2f} mm. '
        'The bound applies to corresponding-pixel RMSE. It explains the limited scope of one scale per view on this '
        'measure. '
        '[Experimental design](../../docs/DEPTH_GEOMETRY_HEADROOM.md) · [Replayable view statistics](geometry_headroom.json)',
        '', '## Which errors do sparse matches cover?', '']
    spatial = read(PUBLIC/'spatial_support.json')['results']['overall'][REFS[0]]
    fit = spatial['fit']
    fit_pixels = sum(pair['near']['pixel_fraction'] for pair in fit.values())
    fit_error = sum(pair['near']['error_fraction'] for pair in fit.values())
    both_error = sum(pair['near']['error_fraction'] for pair in spatial['both'].values())
    boundary = [cell for k,pair in fit.items() if k!='interior' for cell in pair.values()]
    lines += [f'After independently aligning each view, {sum(c["pixel_fraction"] for c in boundary):.2%} of valid '
        f'pixels are near a depth hole or a geometric jump. They carry {sum(c["error_fraction"] for c in boundary):.2%} '
        f'of registered squared error. The interior carries the remainder. The eight-pixel neighborhoods of fitting-fold '
        f'RGB matches cover {fit_pixels:.2%} of valid pixels and {fit_error:.2%} of squared error. Even the union of '
        f'both folds covers only {both_error:.2%} of squared error. These fractions describe directly observed support. '
        'A smooth model can propagate information farther than eight pixels. Whether dense image alignment can identify '
        'a beneficial correction on the remaining surfaces is the next local research question. '
        '[Region definitions](../../docs/DEPTH_SPATIAL_SUPPORT.md) · [Per-view counts and squared errors](spatial_support.json)',
        '', '## Decision and cost','']
    checks = dict(local_gain='At least 5% lower error than the best local control',
        historical_gain='At least 5% lower error than the best historical control',
        component_safety='Neither distance component more than 2% worse than the local control',
        all_scenes='Every scene at least as accurate as the local control',
        released='Released-depth score at least as good as the local control')
    lines.extend(f'- {label} — {"passed" if summary["checks"][k] else "failed"}.' for k,label in checks.items())
    timings = {k:mean(r['seconds'][k] for r in rows) for k in rows[0]['seconds']}
    lines += ['',f'The local correction averaged {timings["guarded_cpu_postprocess"]:.3f} s per five-view request '
        f'on the CPU, including dense point creation. Prior measurements on the same RGB batches averaged '
        f'{timings["instrumented_3080_heads"]:.2f} s for the instrumented frozen RTX 3080 pass and '
        f'{timings["fresh_cpu_matches"]:.2f} s for fresh CPU matching and track export. Their sum, '
        f'{sum(timings.values()):.2f} s, is a component-based estimate. Model initialization, dataset I/O and '
        'offline reference evaluation are excluded. The historical GPU measurements use different hardware. '
        'A head-to-head latency claim would require a common end-to-end benchmark.', '',
        f'Local execution added **$0 RunPod cost**. The two archives contain the same {len(archive["files"])} '
        f'unique point maps, {archive["total_bytes"]:,} bytes per drive. [Prediction seals and timing](predictions.json) · '
        '[Archive checksums](archive.json) · [Budget](../../docs/BUDGET.md)', '', '## Reproduction','',
        'From the repository root, run the following commands with the CPU dependencies used by GitHub Actions. '
        'The second command refits both bounded solvers from the published match systems, checks validation on '
        'the disjoint fold, imports the original controls and rebuilds the tables from the measured dense distances.','',
        '```sh','python -m unittest discover -s scripts -p test_fixed_camera_depth_scale.py',
        'python scripts/score_depth_scale_study.py verify','python scripts/diagnose_depth_scale_pixels.py verify',
        'python scripts/depth_geometry_headroom.py verify','python scripts/depth_spatial_support.py verify','```','',
        'To recompute distances from pixels and depths, use a separate scratch checkout with the '
        'dense pilot RGB and depth inputs and the frozen head archives. Generate predictions into an empty study '
        'output directory with `python scripts/run_depth_scale_study.py`, then run '
        '`python scripts/score_depth_scale_study.py evaluate` in the pinned TCO environment. '
        '[Dense input reconstruction](../dense_focal_pilot/REPLAY.md) · [Frozen head capture](../matched_focal_heads/RESULTS.md) · '
        '[Public sparse match systems](match_systems.zip)', '']
    (PUBLIC/'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))


if __name__=='__main__':
    main()
