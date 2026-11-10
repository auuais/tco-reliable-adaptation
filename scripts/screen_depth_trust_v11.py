"""Frozen, two-phase CPU development screen; seal every candidate before scoring any."""
import argparse, concurrent.futures, csv, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from evaluation_stability import dump,sha
from depth_trust import MODES,move_along_rays
from pointmap_camera_diagnostic import recover

ROOT=Path(__file__).resolve().parents[1];STUDY="depth_trust_v11"
MF=ROOT/"manifests"/(STUDY+".json");RAW=ROOT/"reports"/STUDY;PUB=ROOT/"results"/STUDY
def read(p):return json.loads(p.read_text())
def need(value,message):
    if not value:raise ValueError(message)
def plan():
    source=ROOT/"results/repeat_calibration_v6/per_run.csv"
    with source.open() as f:rows=[r for r in csv.DictReader(f) if r["condition"]=="both" and r["calibration"]=="released"]
    need(len(rows)==13,"Unexpected source cohort")
    return dict(study_id=STUDY,phase="Development only, after inspecting camera/shape diagnosis. No claim of independent data or novelty.",
      sources={k:sha(ROOT/k) for k in ["scripts/screen_depth_trust_v11.py","scripts/depth_trust.py","scripts/pointmap_camera_diagnostic.py","scripts/run_repeat_calibration.py","results/pointmap_precision_v9/summary.json","results/repeat_calibration_v6/per_run.csv"]},
      runs=[dict(run_id=r["run_id"],scene=r["scene"],seed=int(r["seed"]),repeat=int(r["execution_repeat"]),input_points_sha256=r["prediction_file_sha256"]) for r in rows],
      candidates=list(MODES),evaluations=52,offline_baseline="V9 upstream FP32 zero, identical captured heads/unchanged released reference; absolute metrics also disclosed",
      policy="Keep all final cameras and global median log depth ratio. Freeze relative depth change, halve everywhere, or clip to +/-log(1.05). TCO identity is exact. Use all pixels, no references, image correspondences or GT masks for method.",
      comparison="Repeat 0 only, 3 development scenes x 3 seeds. Strongest of unchanged TCO, half-depth, frozen-depth, or FP32 zero by mean normalized score. Max 1% scene-score regression, no increase in harmful cells (either distance >1% above FP32 zero), >=5% relative gain, and absolute gain larger than largest repeated score range of candidate and comparator across Heads & Fire. Do not pool repeat 1/2 in primary means or select best runs.",
      scope="Endpoints only. Does not constrain training or show savings in training. All candidates inherit original 40 TCO updates. Any early screen success still requires noisy-prior controls and independent evidence before spending later funds.",
      threads_per_evaluation=2,max_cpu_workers=3,seed=42,timeout_seconds=600)
def prepare(m,evidence):
    from repeatability_capture import array_digest
    RAW.mkdir(parents=True,exist_ok=False);PUB.mkdir(parents=True,exist_ok=False);records=[]
    for run in m["runs"]:
        rid=run["run_id"];p=evidence/"runs"/rid
        need(sha(p/"predictions.npy")==run["input_points_sha256"],rid);x=np.load(p/"predictions.npy",allow_pickle=False)
        cap=read(p/"capture.json")["records"]["initial_predictions"];need(sha(p/cap["file"])==cap["file_sha256"],rid)
        heads=dict(np.load(p/cap["file"],allow_pickle=False));need(array_digest(heads)==cap["tensor_sha256"],rid)
        cams=[recover(a) for a in x];z=np.stack([h for k,h in cams]);centers=[]
        for cam,_ in cams:
            E=cam["extrinsics"];centers.append(-E[:3,:3].T@E[:3,3])
        for mode in MODES:
            directory=RAW/(rid+"__"+mode);directory.mkdir()
            y,info=move_along_rays(x,centers,heads["depth"][0,...,0],z,mode)
            path=directory/"predictions.npy";np.save(path,y)
            if mode=="tco":need(sha(path)==run["input_points_sha256"],"Identity is not exact")
            records.append(dict(**run,mode=mode,job_id=directory.name,output_sha256=sha(path),**info))
    need(len(records)==52,"Missing predictions")
    dump(PUB/"sealed_predictions.json",dict(manifest_sha256=sha(MF),records=records))
def worker(index,evidence):
    import torch, open3d as o3d
    from run_repeat_calibration import evaluate
    from repeatability_capture import array_digest
    r=read(PUB/"sealed_predictions.json")["records"][index];p=RAW/r["job_id"]
    m=read(MF);np.random.seed(m["seed"]);torch.manual_seed(m["seed"]);o3d.utility.random.seed(m["seed"]);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    target=read(evidence/"runs"/r["run_id"]/"run.json")
    arrays=dict(np.load(evidence/target["evaluation_inputs_relative"],allow_pickle=False));need(array_digest(arrays)==target["evaluation_tensor_sha256"],"Reference mismatch")
    need(sha(p/"predictions.npy")==r["output_sha256"],"Prediction changed")
    pred=np.load(p/"predictions.npy",allow_pickle=False)
    result=evaluate(pred,dict(pointclouds=arrays["reference_points"],images=torch.from_numpy(arrays["images"]),valid_mask=arrays["valid_mask"]),p)
    dump(p/"result.json",dict(job_id=r["job_id"],reference_sha256=target["evaluation_tensor_sha256"],metrics=result))
def score(m,evidence):
    sealed=read(PUB/"sealed_predictions.json")
    need(sealed["manifest_sha256"]==sha(MF) and len(sealed["records"])==52,"Seal mismatch")
    env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",NUMEXPR_NUM_THREADS="2",PYTHONHASHSEED="42")
    def call(i):
        job=sealed["records"][i]["job_id"]
        with (RAW/job/"evaluation.log").open("x") as log:
            subprocess.run([sys.executable,str(Path(__file__)),"--worker",str(i),"--evidence",str(evidence)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for i,_ in enumerate(pool.map(call,range(52))):print("Scored",i+1,"/52",flush=True)
    rows=[];keys=["accuracy_mean_m","completeness_mean_m","normal_consistency_mean"]
    zero={r["scene"]:r for r in read(ROOT/"results/pointmap_precision_v9/summary.json")["rows"] if r["mode"]=="upstream_cpu_float32"}
    for r in sealed["records"]:
        v=read(RAW/r["job_id"]/"result.json");z=zero[r["scene"]]
        need(z["reference_tensor_sha256"]==v["reference_sha256"] and all(np.isfinite(v["metrics"][k]) for k in keys),"Invalid comparison")
        ratios=[v["metrics"][k]/z["metrics"][k] for k in keys[:2]]
        rows.append(dict(**r,metrics={k:v["metrics"][k] for k in keys},score=float(np.mean(ratios)),harmful=max(ratios)>1.01))
    modes={};scenes=list(zero)
    for mode in MODES:
        allmode=[r for r in rows if r["mode"]==mode];primary=[r for r in allmode if r["repeat"]==0]
        need(len(primary)==9 and len({(r["scene"],r["seed"]) for r in primary})==9,"Unpaired primary cases")
        repeats={}
        for s in ["heads/seq-01","fire/seq-03"]:
            vals=[r["score"] for r in allmode if r["seed"]==42 and r["scene"]==s];need(len(vals)==3,"Incomplete repeats")
            repeats[s]=dict(scores=vals,score_range=float(np.ptp(vals)))
        modes[mode]=dict(score=float(np.mean([r["score"] for r in primary])),harmful=sum(r["harmful"] for r in primary),
          scenes={s:float(np.mean([r["score"] for r in primary if r["scene"]==s])) for s in scenes},repeats=repeats,
          metrics={k:float(np.mean([r["metrics"][k] for r in primary])) for k in keys})
    modes["zero"]=dict(score=1.,harmful=0,scenes={s:1. for s in scenes},repeats={},metrics={k:float(np.mean([r["metrics"][k] for r in zero.values()])) for k in keys})
    best=min(["tco","freeze_shape","half_shape","zero"],key=lambda k:modes[k]["score"]);b=modes[best];c=modes["bounded_5pct"]
    gain=b["score"]-c["score"];noise=max(q["score_range"] for a in [c,b] for q in a["repeats"].values())
    gate=dict(gain_at_least_5pct=gain/b["score"]>=.05,harm_not_increased=c["harmful"]<=b["harmful"],no_scene_regression=all(c["scenes"][s]<=1.01*b["scenes"][s] for s in scenes),gain_exceeds_repeat_range=gain>noise)
    dump(PUB/"summary.json",dict(manifest_sha256=sha(MF),sealed_predictions_sha256=sha(PUB/"sealed_predictions.json"),modes=modes,best_simple_baseline=best,relative_gain=gain/b["score"],largest_repeat_range=noise,gate=gate,passes=all(gate.values()),rows=rows,additional_runpod_cost_usd=0,later_funds_released=False))
def main():
    p=argparse.ArgumentParser();p.add_argument("--freeze",action="store_true");p.add_argument("--prepare",action="store_true");p.add_argument("--score",action="store_true");p.add_argument("--worker",type=int);p.add_argument("--evidence",type=Path);a=p.parse_args()
    if a.freeze:
        need(not MF.exists(),"Manifest exists");dump(MF,plan());return
    m=read(MF);need(m==plan(),"Source or plan changed")
    if a.prepare:prepare(m,a.evidence)
    elif a.score:score(m,a.evidence)
    elif a.worker is not None:worker(a.worker,a.evidence)
    else:print("Manifest verified:",sha(MF))
if __name__=="__main__":main()
