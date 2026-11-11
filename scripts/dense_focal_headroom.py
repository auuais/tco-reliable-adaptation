"""Measure how much the two executed focal choices can change reconstruction error."""
import argparse
from statistics import mean

from evaluation_stability import dump, sha
from score_dense_focal_pilot import METRICS, PUBLIC, REFS
from score_saved_geometry import read


def analyze(rows, reference):
    conditions = {}
    for row in rows:
        job = row['job']
        key = (job['scene'], job['case'], job['direction'])
        outcome = row['scores'][reference]
        score = mean(outcome['final'][m] / outcome['initial'][m] for m in METRICS[:2])
        arms = conditions.setdefault(key, {}).setdefault(job['seed'], {})
        if job['arm'] in arms:
            raise ValueError('Repeated arm for one condition and seed')
        arms[job['arm']] = score
    if len(conditions) != 12:
        raise ValueError('Expected twelve camera conditions')
    choices = ('retain', 'consensus')
    entries = []
    for (scene, case, direction), seeds in sorted(conditions.items()):
        if set(seeds) != {42, 123} or any(set(s) != {'retain', 'consensus', 'adaptive'} for s in seeds.values()):
            raise ValueError('Both seeds and three independently executed arms are required')
        physical = 'consensus' if case == 'metadata_fault' else 'retain'
        for seed, other in ((42, 123), (123, 42)):
            scores = seeds[seed]
            oracle = min(choices, key=scores.__getitem__)
            repeat = min(choices, key=seeds[other].__getitem__)
            entries.append(dict(scene=scene, case=case, direction=direction, seed=seed,
                                physical_choice=physical, same_seed_best=oracle, other_seed_best=repeat,
                                tied=scores['retain'] == scores['consensus'],
                                scores=dict(**scores, frozen=1., correct_metadata=scores[physical],
                                            same_seed_minimum=scores[oracle], other_seed_selection=scores[repeat])))
    means = {k: mean(r['scores'][k] for r in entries) for k in entries[0]['scores']}
    baseline = min(('retain','consensus','frozen'), key=means.__getitem__)
    return dict(rows=entries, means=means, best_unconditional=baseline,
                relative_gain_vs_best_unconditional={k:1-s/means[baseline] for k,s in means.items()},
                physical_target_strictly_best=sum(r['physical_choice'] == r['same_seed_best'] and not r['tied'] for r in entries),
                exact_ties=sum(r['tied'] for r in entries),
                conditions_with_same_strict_winner_at_both_seeds=sum(
                    r['same_seed_best'] == r['other_seed_best'] and not r['tied'] and not other['tied']
                    for r, other in zip(entries[::2], entries[1::2])))


def evidence():
    path = PUBLIC / 'summary.json'
    data = read(path)
    refs = {ref: analyze(data['rows'], ref) for ref in REFS}
    for ref in REFS:
        official = data['results']['references'][ref]['overall']
        if any(abs(refs[ref]['means'][k] - official[k]['score']) > 1e-12 for k in official):
            raise ValueError('Unconditional/control scoring differs from the registered endpoint')
    return dict(summary_sha256=sha(path), references=refs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('write', 'verify'))
    args = parser.parse_args()
    value = evidence()
    path = PUBLIC / 'focal_headroom.json'
    if args.action == 'verify':
        if read(path) != value:
            raise ValueError('Focal-choice diagnostic differs from published outcomes')
    elif path.exists():
        raise FileExistsError(path)
    else:
        dump(path, value)
    print('Verified all 24 paired focal choices under both depth references.')
