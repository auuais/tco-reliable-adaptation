"""Exploratory mechanism analysis; does not modify the frozen primary decision gate."""
import argparse
import json
from pathlib import Path
import numpy as np
from calibration_audit import NOMINAL_FOCAL, REGISTRATION_RGB_FOCAL, resized_k
from evaluation_stability import dump

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-root",type=Path,required=True)
    parser.add_argument("--results",type=Path,default=ROOT/"results/repeat_calibration_v6")
    args=parser.parse_args()
    summary=json.loads((args.results/"summary.json").read_text())
    read=lambda run,name:json.loads((args.evidence_root/"runs"/run/name).read_text())
    repeat_analysis=[]
    for group in summary["repeats"]:
        histories=[read(run,"losses.json") for run in group["run_ids"]]
        channels={}
        for key in ["total","pose_rot_loss","pose_trans_loss","intrinsics_loss","mv_consistency_loss"]:
            traces=np.asarray([[step[key] for step in history] for history in histories])
            spread=np.ptp(traces,axis=0)
            different=np.flatnonzero(spread>0)
            beyond=np.flatnonzero(spread>1e-6)
            channels[key]=dict(first_any_difference_step=int(different[0]) if len(different) else None,
                first_difference_over_absolute_1e_6_step=int(beyond[0]) if len(beyond) else None,
                maximum_absolute_range=float(spread.max()),ranges_by_step=spread.tolist())
        repeat_analysis.append(dict(scene=group["scene"],condition=group["condition"],
                                    initial_state_checks=group["all_equal_by_state"],loss_channels=channels))
    settings=dict(released=resized_k(NOMINAL_FOCAL,uniform=True),
                  axis_nominal=resized_k(NOMINAL_FOCAL),
                  axis_registration=resized_k(REGISTRATION_RGB_FOCAL))
    focal_targets={key:k.astype(np.float32)[[0,1],[0,1]] for key,k in settings.items()}
    saturation=[]
    for row in summary["rows"]:
        if row["condition"]!="both" or row["execution_repeat"]!=0:continue
        history=read(row["run_id"],"calibration_residuals.json")
        predicted=np.asarray([step["predicted_focal_xy_pixels"] for step in history],dtype=np.float32)
        slopes={key:np.clip(predicted-target,-1,1) for key,target in focal_targets.items()}
        fractions={key:float(np.mean(np.abs(predicted-target)<1)) for key,target in focal_targets.items()}
        contrasts={key:dict(fraction_of_focal_components_with_changed_slope=float(np.mean(slopes[key]!=slopes["released"])),
            maximum_absolute_slope_difference=float(np.max(np.abs(slopes[key]-slopes["released"]))),
            mean_absolute_slope_difference=float(np.mean(np.abs(slopes[key]-slopes["released"]))))
            for key in ["axis_nominal","axis_registration"]}
        saturation.append(dict(run_id=row["run_id"],scene=row["scene"],seed=row["seed"],
            actual_calibration=row["calibration"],quadratic_fraction_by_counterfactual_target=fractions,
            fixed_prediction_counterfactuals_vs_released=contrasts))
    dump(args.results/"mechanism_diagnostics.json",dict(analysis_type="Exploratory; frozen primary outcomes/gate unchanged",
        repeats=repeat_analysis,focal_saturation=saturation,
        interpretation=["Identical initial tensors and RNG with different later tensors narrows the diagnosis to execution/optimization, not a particular operator.",
            "Counterfactual slopes are evaluated at the SAME observed predicted focal values; they do not simulate an alternate optimization trajectory.",
            "Smooth-L1 focal slope is clip(predicted-target,-1,1); actual loss also applies averaging and weights. Constant principal-point terms have zero model gradient.",
            "Low slope sensitivity can explain why calibration changes provide little local optimization signal, but does not establish causality or a new method."]))
    print(json.dumps(dict(repeat_groups=len(repeat_analysis),calibration_trajectories=len(saturation))))


if __name__=="__main__":main()
