"""Explore neural depth changes beyond a relaxed frozen-canonical alignment family."""
import numpy as np
from score_mast3r_neural_transfer import PUBLIC,RAW,read,sha,dump


def projection(base,target,block=8):
    base,target=np.asarray(base,dtype=float),np.asarray(target,dtype=float)
    assert base.shape==target.shape and np.isfinite(base).all() and np.isfinite(target).all()
    height,width=base.shape
    assert height%block==width%block==0
    tile=lambda x:x.reshape(height//block,block,width//block,block).transpose(0,2,1,3).reshape(-1,block*block)
    b,t=tile(base),tile(target)
    anchor=b[:,(block//2)*block+block//2]
    assert (anchor>0).all()
    scale=float(np.sum(b*t)/np.sum(t*t))
    assert scale>0
    t=scale*t
    x=b/anchor[:,None]-1
    xc=x-x.mean(axis=1,keepdims=True)
    tc=t-t.mean(axis=1,keepdims=True)
    denominator=np.sum(xc*xc,axis=1)
    slope=np.divide(np.sum(xc*tc,axis=1),denominator,out=np.zeros_like(denominator),where=denominator>0)
    fitted=t.mean(axis=1,keepdims=True)+slope[:,None]*xc
    unit=float(np.median(base))
    assert unit>0
    update_energy=float(np.mean((t-b)**2)/unit**2)
    residual_energy=float(np.mean((t-fitted)**2)/unit**2)
    fraction=residual_energy/update_energy if update_energy>1e-24 else None
    if fraction is not None:assert -1e-8<=fraction<=1+1e-8
    return dict(global_scale_removed=scale,relative_rms_update=float(np.sqrt(update_energy)),
                relative_rms_outside_relaxed_family=float(np.sqrt(residual_energy)),
                outside_energy_fraction=fraction)


def run():
    import torch
    seal=read(PUBLIC/'predictions.json');plan=read(PUBLIC/'plan.json')
    assert seal['plan_sha256']==sha(PUBLIC/'plan.json')
    templates={};records=[];controls=[]
    for slug in ('heads','chess','fire'):
        paths=sorted((RAW/'cache'/(slug+'_zero')/'canon_views').glob('*.pth'))
        assert len(paths)==5
        templates[slug]=[(p,torch.load(p,map_location='cpu',weights_only=True)[0][1].numpy()) for p in paths]
    for row in seal['rows']:
        native=PUBLIC/'native_outputs'/(row['tag']+'.npz')
        assert sha(native)==row['native_sha256']
        with np.load(native) as z:depth=z['depth']
        # Canonical paths sort by image hash, not frame order. Recover the exact
        # image-name hashes using the original capture naming convention.
        import hashlib
        names=[f'{row["input_key"]}_{plan["rgb_sha256"][row["scene"]][:12]}_{i}' for i in range(5)]
        bases={p.name.split('_subsample=')[0]:a for p,a in templates[row['scene']]}
        for i,name in enumerate(names):
            base=bases[hashlib.md5(name.encode()).hexdigest()]
            result=projection(base,depth[i])
            record=dict(tag=row['tag'],scene=row['scene'],arm=row['arm'],post=row['post'],view=i,**result)
            records.append(record)
            if row['arm']=='zero':controls.append(result['relative_rms_outside_relaxed_family'])
    assert len(records)==525 and len(controls)==105
    # An empirical positive control validates the source-derived family. It is
    # not a reconstruction-quality or research-advancement threshold.
    passed=max(controls)<=2e-6
    canonical=[]
    for job in plan['jobs']:
        if job['arm']=='zero':continue
        for base_path,base in templates[job['scene']]:
            path=RAW/'cache'/job['case']/'canon_views'/base_path.name
            adapted=torch.load(path,map_location='cpu',weights_only=True)[0][1].numpy()
            canonical.append(dict(case=job['case'],view_hash=base_path.name.split('_subsample=')[0],**projection(base,adapted)))
    matching=[]
    for job in plan['jobs']:
        if job['arm']=='zero':continue
        folder=RAW/'cache'/job['case']/'corres_conf=desc_conf_subsample=8'
        zero=RAW/'cache'/(job['scene']+'_zero')/folder.name
        files=sorted(folder.glob('*.pth'));assert len(files)==20
        pairs=[]
        for path in files:
            # These are this study's locally generated, checksummed cache files.
            old=torch.load(zero/path.name,map_location='cpu',weights_only=False)[1]
            new=torch.load(path,map_location='cpu',weights_only=False)[1]
            def keys(value):
                xy=np.column_stack([np.asarray(value[0]),np.asarray(value[1])])
                integer=xy.astype(np.int64);np.testing.assert_array_equal(xy,integer)
                return np.unique(np.ascontiguousarray(integer).view(np.dtype((np.void,32))).ravel())
            a,b=keys(old),keys(new)
            overlap=len(np.intersect1d(a,b));union=len(a)+len(b)-overlap
            pairs.append(dict(file=path.name,old_count=len(a),new_count=len(b),shared_count=overlap,
                              jaccard=overlap/union if union else 1.))
        matching.append(dict(case=job['case'],mean_directed_match_jaccard=float(np.mean([p['jaccard'] for p in pairs])),pairs=pairs))
    dump(PUBLIC/'depth_family_audit.json',dict(seal_sha256=sha(PUBLIC/'predictions.json'),
        exploratory=True,positive_control_max_relative_rms=max(controls),positive_control_pass=passed,
        final_depth_records=records,canonical_records=canonical,matching_records=matching,
        scope='Within-view axial-depth representability in a relaxed two-coefficient-per-cell family after removing global scale. Not a full world-geometry nullspace or accuracy test.'))
    print('Depth-family audit:',len(records),'final views;',len(canonical),'canonical changes; zero-control pass:',passed)
    if not passed:raise ValueError('Frozen-canonical family positive control failed; do not interpret neural residuals')


if __name__=='__main__':run()
