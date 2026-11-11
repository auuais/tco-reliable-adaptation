"""Write the measured depth-warp study tables from the public evidence."""
from statistics import mean

from run_dense_rgb_depth import PARENT, PUBLIC
from score_dense_focal_pilot import METRICS, REFS
from score_depth_scale_study import HISTORICAL
from score_saved_geometry import read


NAMES = dict(frozen='Frozen VGGT, local FP32',direct='Direct focal substitution',shared='One depth scale',
    previous_guard='Guarded per-view scales',two_source='RGB, two sources',four_source='RGB, four sources',
    guarded='RGB, two sources with independent validation',native_3090='Frozen VGGT, archived RTX 3090',
    tco_retain='TCO, retain',tco_consensus='TCO, consensus',tco_adaptive='TCO, adaptive')


def effect(gain):
    return f'{abs(gain):.2%} '+('lower' if gain>=0 else 'higher')


def main():
    scores = read(PUBLIC/'scores.json')['results']
    pixel = read(PUBLIC/'pixel_check.json')['results']
    positive = read(PUBLIC/'positive_control.json')['results']['overall']
    pred = read(PUBLIC/'predictions.json')['batches']
    rows = read(PARENT/'predictions.json')['rows']
    main_ref,secondary = (scores['references'][ref]['overall'] for ref in REFS)
    result = 'passed' if scores['advance'] else 'missed'
    lines = ['# Dense RGB depth correction with an independent source-view check','',
        f'The guarded method had **{effect(scores["local_gain"])}** registered 3D error than the best local control '
        f'({NAMES[scores["local_comparator"]]}) and **{effect(scores["historical_gain"])}** error than the best historical '
        f'control ({NAMES[scores["historical_comparator"]]}). It {result} the registered development gate. '
        '[Prospective design](../../docs/DENSE_RGB_DEPTH.md) · [Raw surface scores and decision](scores.json)','',
        '## Surface reconstruction','',
        'Each condition has equal weight. The primary score averages the ratios of accuracy and completeness to the '
        'local frozen FP32 model. Smaller values indicate smaller distances. Normal consistency is higher when normals agree. '
        'The historical TCO arms average the original two seeds. These are three development sequences, nine distinct RGB '
        'batches and twelve controlled cases. The two signs of a metadata fault share the same RGB images.','',
        '| Method | Registered score | Accuracy (mm) | Completeness (mm) | Normal consistency | Released score |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,label in NAMES.items():
        raw=main_ref[arm]['raw']
        lines.append(f'| {label} | {main_ref[arm]["score"]:.4f} | {1000*raw[METRICS[0]]:.2f} | '
                     f'{1000*raw[METRICS[1]]:.2f} | {raw[METRICS[2]]:.4f} | {secondary[arm]["score"]:.4f} |')
    lines += ['', '| Scene | Guarded score | Difference from best local control | All four source views support a patch |',
        '|---|---:|---:|---:|']
    for scene,data in scores['references'][REFS[0]]['scenes'].items():
        fraction = mean(v['joint_pixels']/(392*518) for r in rows if r['scene']==scene for v in pred[r['input_key']]['views'])
        lines.append(f'| {scene} | {data["guarded"]["score"]:.4f} | {effect(scores["gain_by_scene"][scene])} | {fraction:.2%} |')
    lines += ['', '## Do the changes move pixels closer to the reference?', '',
        'All methods use the baseline global similarity for this diagnostic. An oracle chooses the best of the same five '
        'candidate depths directly from the reference at each pixel. It measures the available range of corrections. '
        'The inference methods select depths using only RGB, frozen depth and frozen cameras.','',
        '| Method | Changed valid pixels | Improved among changed | Mean RMSE (mm) | Squared-error reduction outside sparse support |',
        '|---|---:|---:|---:|---:|']
    for arm in ('two_source','four_source','guarded','oracle'):
        a = pixel[REFS[0]]['all'][arm]
        far = pixel[REFS[0]]['far'][arm]
        lines.append(f'| {NAMES.get(arm,"Reference oracle")} | {a["mean_changed_fraction"]:.2%} | '
                     f'{a["pooled_improved_fraction"]:.2%} | {1000*a["mean_rmse_m"]:.2f} | {far["mean_sse_reduction"]:.2%} |')
    oracle=pixel[REFS[0]]['all']['oracle']
    lines += ['',f'Selecting the best available depth using reference geometry lowered mean corresponding-pixel RMSE from '
        f'{1000*oracle["baseline_mean_rmse_m"]:.2f} to {1000*oracle["mean_rmse_m"]:.2f} mm. These values concern '
        'the corresponding-pixel metric and do not give a ceiling on nearest-neighbor surface error. '
        'Sparse support means within eight pixels of a fitting-fold RGB match. The proportions of changed pixels are '
        'averaged by condition. The fraction improved among changes pools the changed pixels. '
        '[Pixel measurements for both references](pixel_check.json)','',
        '## Positive control with known cameras and axial depth','',
        'Only this diagnostic uses reference cameras and source depths. The target starts at its true depth multiplied '
        'by exp(−0.08), 1 or exp(+0.08). Here “true” denotes the reference depth used to inject the error. RGB-D noise, '
        'resampling, registration error and changing appearance can still affect agreement between images. '
        'The clean column counts false changes over all valid pixels. '
        'The other fractions distinguish the area with test support from exact recovery on that area.','',
        '| Rule | Short start eligible | Short exactly recovered among eligible | Clean falsely changed | Long start eligible | Long exactly recovered among eligible |',
        '|---|---:|---:|---:|---:|---:|']
    def pct(v):
        return 'no eligible pixels' if v is None else f'{v:.2%}'
    for a in ('nearest','two_source','four_source','guarded'):
        s,c,l=(positive[key][a] for key in ('short','clean','long'))
        values=(s['mean_eligible_fraction'],s['pooled_exact_among_eligible'],1-c['mean_exact_fraction'],
                l['mean_eligible_fraction'],l['pooled_exact_among_eligible'])
        lines.append('| '+NAMES.get(a,'Nearest source')+' | '+' | '.join(pct(v) for v in values)+' |')
    lines += ['', 'Eligible fractions average the twelve conditions. Exact recovery among eligible pixels pools their counts. '
        'The original rule requires all five hypotheses to pass the source-depth visibility check. For parallel cameras '
        'viewing a front-facing plane at depth z, a long initial depth of exp(0.08)z has an extreme candidate of '
        'exp(0.16)z, about 1.174z. That candidate exceeds the 1.10z visibility limit and can exclude the pixel even when '
        'the correct depth is one of the other four candidates. '
        'The clean result also exposes a selection problem among eligible pixels. The guarded rule changes '
        f'{1-positive["clean"]["guarded"]["pooled_exact_among_eligible"]:.1%} of them even at the reference depth. '
        'The next local comparison will separate this photometric ambiguity from the visibility restriction by testing '
        'source-depth agreement and visibility anchored at the starting depth. '
        '[Positive-control protocol](../../docs/PHOTOMETRIC_POSITIVE_CONTROL.md) · [Counts by scene, start and rule](positive_control.json)', '',
        '## Runtime, evidence and reproduction','']
    time = mean(pred[r['input_key']]['selection_seconds']+pred[r['input_key']]['point_construction_seconds']['guarded'] for r in rows)
    files = read(PUBLIC/'archive.json')
    lines += [f'The local CPU took {time:.2f} seconds per five-view condition for the RGB scoring, rule selection and '
        'guarded point construction, using cached frozen heads. This timing excludes model inference, file I/O and '
        'reference evaluation. Historical TCO ran on another GPU. The new experiment added **$0** to the RunPod bill. '
        f'Two physical drives hold verified copies of all {len(files["files"])} distinct point maps '
        f'({files["total_bytes"]:,} bytes per copy). [Archive checksums](archive.json) · '
        '[Per-pixel choices](choices.zip) · [Input and output hashes](predictions.json)','',
        'Run the following from the repository root with the CPU validation dependencies. The public checks replay '
        'the discrete predictions, metric tables and pixel counts. Full image-based reproduction also needs the 7-Scenes '
        'inputs, frozen heads and offline reference maps described in the parent experiment. In a scratch checkout '
        'with an empty output folder for this study, run the inference script first, followed by each evaluator in '
        'evaluate mode. [Parent inputs and evaluation](../fixed_camera_depth_scale/RESULTS.md)','',
        '```sh','python -m unittest discover -s scripts -p test_dense_rgb_depth.py','python scripts/score_dense_rgb_depth.py verify',
        'python scripts/dense_rgb_pixel_check.py verify','python scripts/photometric_positive_control.py verify','```','']
    (PUBLIC/'RESULTS.md').write_bytes('\n'.join(lines).encode())


if __name__=='__main__':
    main()
