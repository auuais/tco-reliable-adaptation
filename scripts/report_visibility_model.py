"""Write the visibility factorial, real model comparison and spending decision."""
from statistics import mean

from report_dense_rgb_depth import NAMES as OLD_NAMES,effect
from run_dense_rgb_depth import PARENT
from run_visibility_model import PUBLIC
from score_dense_focal_pilot import METRICS,REFS
from score_saved_geometry import read
from visibility_depth_evidence import MODEL_ARMS


NAMES = dict(OLD_NAMES,photo_guarded='Previously tested four-view RGB check',guarded='Depth, two supporting views',
             single_depth='Depth, one supporting view',checked_rgb='RGB, two supporting views',
             four_view_depth='Depth, fixed four views',all_candidate_depth='Depth, two views, all candidates visible')


def main():
    control = read(PUBLIC/'positive_control.json')['results']['overall']
    scores = read(PUBLIC/'scores.json')['results']
    pixels = read(PUBLIC/'pixel_check.json')['results']
    seal = read(PUBLIC/'predictions.json')['batches']
    rows = read(PARENT/'predictions.json')['rows']
    reg,release = (scores['references'][ref]['overall'] for ref in REFS)
    winner = NAMES[scores['local_comparator']]
    passed = 'passed' if scores['advance'] else 'missed'
    lines = ['# Can agreement between predicted depth maps refine a frozen reconstruction?','',
        f'The two-view depth check had **{effect(scores["local_gain"])}** registered surface error than the best local '
        f'control ({winner}) and **{effect(scores["historical_gain"])}** error than the best historical control '
        f'({NAMES[scores["historical_comparator"]]}). It {passed} the prospective 5% development threshold. '
        '[Registered design](../../docs/VISIBILITY_DEPTH_EVIDENCE.md) · [Measured surface scores](scores.json)','',
        'Pixelwise source selection and consistency across depth maps are established multi-view stereo techniques. '
        'TCO already uses photometric or geometric compatibility while adapting a shared network. Our controlled '
        'comparison tests how a specific visibility rule in our local depth search changes which errors can be observed, '
        'and how results change when the supporting views use predictions from the same frozen model. '
        '[Schönberger et al., ECCV 2016](https://demuc.de/papers/schoenberger2016mvs.pdf) · '
        '[MVSNet, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf) · '
        '[TCO, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Learning_3D_Reconstruction_with_Priors_in_Test_Time_CVPR_2026_paper.pdf)','',
        '## Known depth errors with reference source geometry','',
        'These tests supply the reference camera and source depths as a positive control. A short target starts at '
        'exp(−0.08) times reference depth and a long target at exp(+0.08) times reference depth. Exact recovery returns '
        'to the reference depth. The clean false-edit rate includes all valid target pixels, including abstentions. '
        'All twelve conditions receive equal weight. The final two columns pool the eligible pixels.','',
        '| Cue and view rule | Short exactly recovered, all pixels | Long exactly recovered, all pixels | Clean false edit | Short exact, eligible | Long exact, eligible |',
        '|---|---:|---:|---:|---:|---:|']
    for label,key in [('RGB, four fixed views, all candidates visible','all_candidates_rgb_four_view'),
                      ('RGB, four fixed views, visibility at starting depth','starting_depth_rgb_four_view'),
                      ('RGB, two supporting views, visibility at starting depth','starting_depth_rgb_two_witness'),
                      ('Depth, two supporting views, all candidates visible','all_candidates_depth_two_witness'),
                      ('Depth, two supporting views, visibility at starting depth','starting_depth_depth_two_witness'),
                      ('Depth, first supporting view, visibility at starting depth','starting_depth_depth_single')]:
        short,clean,long = (control[state][key] for state in ('short','clean','long'))
        values = [short['mean_exact_fraction'],long['mean_exact_fraction'],1-clean['mean_exact_fraction'],
                  short['pooled_exact_among_eligible'],long['pooled_exact_among_eligible']]
        lines.append('| '+label+' | '+' | '.join(f'{v:.2%}' for v in values)+' |')
    short,long = (control[k]['starting_depth_depth_two_witness'] for k in ('short','long'))
    lines += ['',f'The primary rule had two eligible source views for {short["mean_eligible_fraction"]:.2%} of short-start '
        f'pixels and {long["mean_eligible_fraction"]:.2%} of long-start pixels. Exact recovery conditional on support '
        'should be read alongside the smaller absolute recovery fractions. The original four-view RGB control reproduces '
        'every published histogram. [Full 12-way factorial, including every scene](positive_control.json)','',
        '## Frozen model inputs and dense surface evaluation','',
        'Here the model supplies the camera poses and source depths. All methods share the original local FP32 denominator '
        'and twelve development conditions. Smaller distance scores are better. Registered accuracy and completeness are '
        'nearest-neighbor distances in millimeters after the published alignment. Normal consistency is higher when '
        'the two surfaces agree.','',
        '| Method | Registered score | Accuracy (mm) | Completeness (mm) | Normal consistency | Released score |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ('frozen','direct','shared','previous_guard','two_source','four_source','photo_guarded',
                'single_depth','checked_rgb','four_view_depth','all_candidate_depth','guarded',
                'native_3090','tco_retain','tco_consensus','tco_adaptive'):
        m = reg[arm]['raw']
        lines.append(f'| {NAMES[arm]} | {reg[arm]["score"]:.4f} | {1000*m[METRICS[0]]:.2f} | {1000*m[METRICS[1]]:.2f} | '
                     f'{m[METRICS[2]]:.4f} | {release[arm]["score"]:.4f} |')
    lines += ['', '| Development scene | Guarded score | Difference from the strongest aggregate local control | Primary rule eligible, full raster |',
              '|---|---:|---:|---:|']
    for s,d in scores['references'][REFS[0]]['scenes'].items():
        support=mean(v['eligible']['guarded']/(392*518) for r in rows if r['scene']==s for v in seal[r['input_key']]['views'])
        lines.append(f'| {s} | {d["guarded"]["score"]:.4f} | {effect(scores["gain_by_scene"][s])} | {support:.2%} |')
    lines += ['', '## Do the local updates point in the right direction?', '',
        'These comparisons use the same fixed alignment as the direct-focal predictions. The oracle chooses among '
        'the same five depth values using the corresponding reference pixel. It has no access restriction and reproduces '
        'the previous depth-candidate diagnostic. The final column excludes the eight-pixel neighborhoods of sparse '
        'fitting matches. Positive values indicate reduced squared error.','',
        '| Method | Changed valid pixels | Improved among changed | Registered RMSE (mm) | Squared-error reduction outside sparse support |',
        '|---|---:|---:|---:|---:|']
    for arm in tuple(MODEL_ARMS)+('oracle',):
        v = pixels[REFS[0]]['all'][arm]
        far = pixels[REFS[0]]['far'][arm]
        lines.append(f'| {NAMES.get(arm,"Reference oracle")} | {v["mean_changed_fraction"]:.2%} | {v["pooled_improved_fraction"]:.2%} '
                     f'| {1000*v["mean_rmse_m"]:.2f} | {far["mean_sse_reduction"]:.2%} |')
    checks = ', '.join(f'{label} {"passed" if scores["checks"][k] else "failed"}' for k,label in
        [('local_gain','local gain'),('historical_gain','historical gain'),('component_safety','component limits'),
         ('all_scenes','scene criterion'),('released','released-depth check')])
    timing = mean(seal[r['input_key']]['selection_seconds']+seal[r['input_key']]['point_construction_seconds']['guarded'] for r in rows)
    inventory = read(PUBLIC/'archive.json')
    lines += ['',f'The registered checks gave the following outcomes. {checks.capitalize()}. The nine RGB batches come '
        'from three previously examined 7-Scenes sequences. Relative rankings in this development study cannot establish '
        'performance on other scenes, datasets or backbones. [Condition-level pixel evidence](pixel_check.json)']
    transplant = read(PUBLIC/'source_transplant.json')
    source = transplant['results']['overall']
    lines += ['','## What changes when predicted sources replace reference depth?', '',
        'Each row uses the same four target-depth states and the same starting-depth visibility and two-view check. '
        'The first two numerical columns use the intersection of pixels eligible under all four camera/source combinations. '
        'The natural column measures whether a change moves scaled model axial depth closer to the registered reference. '
        'One reference-derived scale per batch converts arbitrary model units to meters. These are diagnostic interventions.', '',
        '| Camera rig | Source depths | Short exact on common pixels | Long exact on common pixels | Natural improved among edits on common pixels | Natural log-depth MSE reduction, all valid pixels |',
        '|---|---|---:|---:|---:|---:|']
    def percentage(x):
        return 'no edits' if x is None else f'{x:.2%}'
    for cell in ('reference_reference','reference_frozen','frozen_reference','frozen_frozen'):
        rig,depth = cell.split('_')
        ratio = source['natural'][cell]['all']
        gain = 1-ratio['final_mse']/ratio['initial_mse']
        row = [source['short'][cell]['common']['exact_among_eligible'],
               source['long'][cell]['common']['exact_among_eligible'],
               source['natural'][cell]['common']['improved_among_changed'],gain]
        lines.append(f'| {rig.capitalize()} | {depth.capitalize()} | '+' | '.join(percentage(x) for x in row)+' |')
    rates = {cell:source['natural'][cell]['common']['improved_among_changed'] for cell in source['natural']}
    grouped = [[view['natural']['frozen_frozen'] for view in transplant['batches'][row['input_key']]['views']] for row in rows]
    common_fraction = mean(sum(c['common']['n'] for c in g)/sum(c['all']['n'] for c in g) for g in grouped)
    reducible = mean(sum(c['common']['initial_sse']-c['common']['oracle_sse'] for c in g)/
                     sum(c['all']['initial_sse']-c['all']['oracle_sse'] for c in g) for g in grouped)
    lines += ['', f'On the common eligible pixels of natural targets, the share of changes that helped was '
        f'{rates["reference_reference"]:.1%} with the reference rig and source depths. Replacing only source depths '
        f'with frozen predictions reduced that share to {rates["reference_frozen"]:.1%}. Replacing only the camera '
        f'rig gave {rates["frozen_reference"]:.1%}. With both predicted inputs it was {rates["frozen_frozen"]:.1%}. '
        'Each rate conditions on the pixels the corresponding rule actually changes. The eligible region is shared, '
        f'but those changed subsets can differ. The common region contains {common_fraction:.2%} of valid natural-target '
        f'pixels and only {reducible:.2%} of the squared log-depth error that the five depth candidates could remove '
        'with reference access, averaging condition-level fractions.', '',
        'The common intersection is recomputed for each target starting condition, since it affects projection '
        'and visibility. Percentages conditional on eligibility or a change pool pixels across the twelve conditions. '
        'Mean squared log-depth error gives conditions equal weight and uses a different gauge and metric from the surface '
        'table. [Source-replacement protocol](../../docs/DEPTH_SOURCE_TRANSPLANT.md) · [Counts, MSE and scene results](source_transplant.json)',
        '', '## Can the depth candidates produce a better surface?', '',
        'The following three point maps receive the registered reference at decision time. All candidate choices '
        'minimize corresponding-pixel squared error at fixed alignment, then the original surface evaluator realigns '
        'and scores each constructed map. The same map is used for both depth references.', '',
        '| Reference-assisted selection | Registered surface score | Released surface score |',
        '|---|---:|---:|']
    grid = read(PUBLIC/'grid_surface.json')['results']
    for arm,label in [('all_pixels','Any candidate at any valid pixel'),('proposal_sites','Any candidate where the first source proposes a change'),
                      ('accept_or_keep','Accept or reject the first source’s actual proposal')]:
        lines.append(f'| {label} | {grid[REFS[0]][arm]["overall"]["score"]:.4f} | {grid[REFS[1]][arm]["overall"]["score"]:.4f} |')
    lines += ['', 'Only the inference results in the first table of surface scores govern spending. A reference-assisted '
        'surface result witnesses what that particular selection attained. It supplies no global optimum for the '
        'nearest-neighbor metric. [Surface diagnostic design](../../docs/DEPTH_GRID_SURFACE_DIAGNOSTIC.md) · '
        '[Scores and per-scene tables](grid_surface.json)','', '## Does a correction transfer beyond the observed pixels?', '',
        'Fit one bounded depth scale per tile on alternating 16×16-pixel training blocks and evaluate it on the other '
        'block color, swapping roles for the second pass. The evaluation pixels never received a first-source proposal. '
        'All four fitting conditions share these exact test pixels. For the random control, the number of training '
        'labels in each tile matches the observed source-proposal sites. Positive values indicate lower '
        'corresponding-pixel squared error on the registered reference.','',
        '| Tiles per image | Reference at all training pixels | Reference at proposal sites | Reference at matched-count random sites | Actual source proposals |',
        '|---|---:|---:|---:|---:|']
    spatial = read(PUBLIC/'spatial_transfer.json')['results']
    for g,methods in spatial['overall'][REFS[0]].items():
        vals = [methods[a]['sse_reduction'] for a in ('all_reference','proposal_reference','random_reference','proposal_value')]
        lines.append(f'| {g}×{g} | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    values = spatial['overall'][REFS[0]]['4']
    lines += ['',f'The test covers {spatial["primary_valid_fraction"]:.2%} of valid pixels. At the registered 4×4 '
        f'resolution, exact reference labels at proposal sites produced '
        f'{effect(values["proposal_reference"]["sse_reduction"])} error than identity on the withheld pixels. '
        f'The same number of random reference labels per tile gave {effect(values["random_reference"]["sse_reduction"])} error. '
        'The locations where the first source asks for a correction therefore transfer poorly to the rest of the tile '
        'on these development sequences. A subsequent local test should distinguish bias from source visibility '
        'from bias introduced by using only nonzero proposals. '
        '[Cross-fit protocol](../../docs/SPATIAL_DEPTH_TRANSFER.md) · [Both references and scene breakdowns](spatial_transfer.json)',
        '', '## Runtime and replay','',
        f'Calculating the full factorial and constructing the primary point map took {timing:.2f} seconds per condition '
        'on average on the local CPU after the original frozen pass. This includes the unused comparison arms and is '
        'not a standalone deployment latency measurement. Cloud cost was **$0**. '
        f'Both physical drives contain {len(inventory["files"])} unique prediction files ({inventory["total_bytes"]:,} bytes '
        'per drive). [Prediction checksums](archive.json) · [Dense corrections](choices.zip) · [Per-view counts and timing](predictions.json)','',
        'With the project CPU dependencies, the following commands check the numerical controls, choices and reported '
        'metrics. Full replay uses the public data and prepared inputs from the previous study. In a scratch checkout '
        'with an empty output folder for this experiment, run `visibility_positive_control.py evaluate`, '
        '`run_visibility_model.py infer`, `score_visibility_model.py evaluate` and `score_visibility_model.py pixels` '
        'from the scripts directory or using the scripts path from the repository root. '
        '[Parent experiment and data](../dense_rgb_depth/RESULTS.md)','',
        '```sh','python -m unittest discover -s scripts -p test_visibility_depth_evidence.py',
        'python scripts/visibility_positive_control.py verify','python scripts/run_visibility_model.py verify',
        'python scripts/score_visibility_model.py verify','python scripts/depth_source_transplant.py verify',
        'python scripts/depth_grid_surface.py verify','python scripts/spatial_depth_transfer.py verify','```','']
    (PUBLIC/'RESULTS.md').write_bytes('\n'.join(lines).encode())


if __name__=='__main__':
    main()
