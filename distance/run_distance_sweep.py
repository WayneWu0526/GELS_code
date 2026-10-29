#!/usr/bin/env python3
"""Run stationary point-dipole simulations from frozen geometric and moment inputs.

All field samples and localization results produced here are synthetic.
The source moment fit is supplied as fixed parameters, not refitted at runtime.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=jsonable, allow_nan=False) + "\n")


def save_csv(path, rows):
    if not rows:
        raise ValueError("no rows for " + str(path))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    repo = Path(__file__).resolve().parent
    base = repo / "reference"
    sys.path.insert(0, str(base))
    from simulation_helpers import (
        MaPS_Estimator,
        rotation_error_deg, affine_weights, constraint_matrix,
    )
    from simulation_core import synthesize_measurements
    from mag_dipole_model import mag_dipole_model

    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    inp = out / "inputs"
    inp.mkdir()
    count = 5 if args.smoke else cfg["distance_count"]
    trials = 12 if args.smoke else cfg["realizations"]
    cfg["effective_distance_count"] = count
    cfg["effective_realizations"] = trials
    save_json(out / "config_snapshot.json", cfg)
    np.seterr(divide="raise", invalid="raise", over="raise")

    parameters_path = repo / cfg["simulation_parameters"]
    parameters = json.loads(parameters_path.read_text())
    save_json(inp / "simulation_parameters.json", parameters)
    full_D = np.asarray(parameters["sensor_coordinates_m"], dtype=float)
    configs = {"Config2": [1,3,4,6,7,9,10,12], "Config3": [2,3,12]}
    Ds = {key: full_D[:, np.array(ids)-1] for key,ids in configs.items()}
    cs = np.asarray(parameters["source_positions_m"], dtype=float)
    axes = np.asarray(parameters["source_axes"], dtype=float)
    nominal_m = parameters["nominal_equal_moment_magnitude_A_m2"]
    fitted = np.asarray(parameters["fitted_signed_moments_A_m2"], dtype=float)
    moment_sets = {"nominal20G": -np.ones(3)*nominal_m, "field_fit": fitted}
    records = [{"index": r["index"], **{k:np.asarray(r[k],dtype=float)
                for k in ("p","R","cs","axes")}}
               for r in parameters["reference_geometries"]]

    # Select the source-plane normal pointing toward the anchor. Every source
    # has the same projection onto this normal, ensuring dr_k/ds > 0.
    anchor = next(r for r in records if r["index"] == parameters["ray_anchor_index"])
    centroid = np.mean(cs,axis=0)
    normal = np.cross(cs[1]-cs[0],cs[2]-cs[0]); normal/=np.linalg.norm(normal)
    if np.dot(anchor["p"]-centroid,normal)<0:
        normal=-normal
    projections=(anchor["p"]-cs)@normal
    if np.min(projections)<=0:
        raise AssertionError("ray is not receding from every source")
    first_distance=np.mean(np.linalg.norm(anchor["p"]-cs,axis=1))*1000
    desired=np.linspace(first_distance,cfg["maximum_mean_source_distance_mm"],count)
    ray=[]
    for j,d in enumerate(desired):
        lo,hi=0.0,2.0
        for _ in range(60):
            s=(lo+hi)/2
            dist=np.mean(np.linalg.norm(anchor["p"]+s*normal-cs,axis=1))*1000
            if dist<d:lo=s
            else:hi=s
        p=anchor["p"]+(lo+hi)/2*normal
        ray.append(dict(index=j,p=p,R=anchor["R"],cs=cs,axes=axes))
    all_dist=np.array([np.linalg.norm(r["p"]-cs,axis=1) for r in ray])
    if not np.all(np.diff(all_dist,axis=0)>0):
        raise AssertionError("source distance is not monotonic")
    save_csv(out / "ray_geometry.csv", [dict(point=r["index"],x_m=r["p"][0],y_m=r["p"][1],z_m=r["p"][2],
        distance_mm=np.mean(ds)*1000,source1_distance_mm=ds[0]*1000,source2_distance_mm=ds[1]*1000,
        source3_distance_mm=ds[2]*1000) for r,ds in zip(ray,all_dist)])

    # Validate analytic dipole gradients independently by finite differences.
    gradient_fd_error=0.0
    for rec in (ray[0],ray[-1]):
        for k in range(3):
            m=axes[k]*fitted[k]
            _, A=mag_dipole_model(rec["p"],m,cs[k],order=1)
            h=1e-6
            fd=np.column_stack([(mag_dipole_model(rec["p"]+np.eye(3)[j]*h,m,cs[k],order=1)[0]
                 -mag_dipole_model(rec["p"]-np.eye(3)[j]*h,m,cs[k],order=1)[0])/(2*h) for j in range(3)])
            gradient_fd_error=max(gradient_fd_error,np.linalg.norm(fd-A)/np.linalg.norm(A))
    if gradient_fd_error>1e-7:
        raise AssertionError("analytic gradient failed finite-difference check")

    summaries=[]
    trial_path=out/"simulation_trials.csv.gz"
    trial_handle=gzip.open(trial_path,"wt",newline="")
    writer=csv.writer(trial_handle)
    writer.writerow(["domain","point","configuration","moment_model","condition","realization","success",
        "distance_mm","position_error_mm","orientation_error_deg","gradient_mean_mT_per_m","gradient_relative_error"])

    # Common random numbers preserve matched conditions. Hardware discrepancy
    # patterns stay fixed along the full ray. Background noise is shared by slots.
    rng=np.random.default_rng(cfg["random_seed"])
    raw_noise=rng.normal(size=(trials,4,3,12))*cfg["raw_slot_noise_std_uT"]*1e-6
    cv_noise=raw_noise[:,:3]-raw_noise[:,3:4]
    pattern=rng.normal(size=(trials,3,3,12))
    pattern-=np.mean(pattern,axis=3,keepdims=True)
    pattern*=cfg["effective_source_residual_vector_rms_uT"]*1e-6/np.sqrt(np.mean(np.sum(pattern**2,axis=2),axis=2))[:,:,None,None]

    for domain, poses in (("recorded",records),("ray",ray)):
        for cname, ids in configs.items():
            D=Ds[cname];idx=np.array(ids)-1
            for model, amps in moment_sets.items():
                for j,rec in enumerate(poses):
                    p,R,srcpos=rec["p"],rec["R"],rec["cs"]
                    moments=rec["axes"]*amps[:,None]
                    F=np.array(synthesize_measurements(D,srcpos,moments,p,R,1))
                    src=[{"p_Ci":v} for v in srcpos]
                    _,_,clean_details=MaPS_Estimator(D,src,list(F))
                    clean_X=np.array(clean_details["X_hat_locals"])
                    true_X=np.array([R.T@mag_dipole_model(p,moments[k],srcpos[k],order=1)[1]@R for k in range(3)])
                    true_norm=np.linalg.norm(true_X)
                    distance=np.mean(np.linalg.norm(p-srcpos,axis=1))*1000
                    for condition in ("ideal","noise","noise_and_residual"):
                        n=1 if condition=="ideal" else trials
                        eps=[];ers=[];gs=[];rel=[];noise_sq=[];failures=0
                        for t in range(n):
                            noise=np.zeros_like(F) if condition=="ideal" else cv_noise[t][:,:,idx]
                            bias=pattern[t][:,:,idx] if condition=="noise_and_residual" else np.zeros_like(F)
                            # Remove the selected subset common mode to use the same
                            # vector-RMS residual convention for Config. 2 and 3.
                            if condition=="noise_and_residual":
                                bias=bias-np.mean(bias,axis=2,keepdims=True)
                                bias=bias*(cfg["effective_source_residual_vector_rms_uT"]*1e-6/np.sqrt(np.mean(np.sum(bias*bias,axis=1),axis=1)))[:,None,None]
                            try:
                                Re,pe,details=MaPS_Estimator(D,src,list(F+noise+bias))
                                X=np.array(details["X_hat_locals"])
                                ep=np.linalg.norm(pe-p)*1000;er=rotation_error_deg(Re,R)
                                g=np.mean(np.linalg.norm(X,axis=(1,2)))*1000
                                re=np.linalg.norm(X-true_X)/true_norm
                                if not np.all(np.isfinite([ep,er,g,re])):raise FloatingPointError("nonfinite output")
                                # The linear gradient estimator gives random-noise
                                # RMS without subtracting the fixed residual model.
                                if condition=="ideal":noise_X=np.zeros_like(X)
                                else:
                                    # Precompute gradient projection only once per
                                    # configuration below via a cache.
                                    if cname not in linear_ops:
                                        w=affine_weights(D);Pw=np.eye(D.shape[1])-np.outer(w,np.ones(D.shape[1]))
                                        linear_ops[cname]=(np.linalg.pinv(constraint_matrix(D)),Pw)
                                    pinv,Pw=linear_ops[cname]
                                    vals=np.array([pinv@(N@Pw).reshape(-1,order="F") for N in noise])
                                    noise_X=np.array([[[v[0],v[1],v[2]],[v[1],v[3],v[4]],[v[2],v[4],-v[0]-v[3]]] for v in vals])
                                eps.append(ep);ers.append(er);gs.append(g);rel.append(re)
                                noise_sq.append(np.linalg.norm(noise_X)**2)
                                writer.writerow([domain,rec["index"],cname,model,condition,t,1,distance,ep,er,g,re])
                            except (ValueError,FloatingPointError,np.linalg.LinAlgError):
                                failures+=1
                                writer.writerow([domain,rec["index"],cname,model,condition,t,0,distance,"","","",""])
                        row=dict(domain=domain,point=rec["index"],configuration=cname,moment_model=model,condition=condition,
                                 distance_mm=distance, trials=n, numerical_failures=failures,
                                 field_mean_G=np.mean(np.linalg.norm(F,axis=1))*1e4,
                                 true_gradient_mean_mT_per_m=np.mean(np.linalg.norm(true_X,axis=(1,2)))*1000,
                                 finite_aperture_relative_gradient_error=np.linalg.norm(clean_X-true_X)/true_norm)
                        for key,values in (("position_error_mm",eps),("orientation_error_deg",ers),("gradient_mean_mT_per_m",gs),("gradient_relative_error",rel)):
                            for label,q in (("p10",10),("median",50),("p90",90),("p95",95)):
                                row[key+"_"+label]=float(np.percentile(values,q)) if values else ""
                        noise_rms=math.sqrt(float(np.mean(noise_sq))) if noise_sq else 0
                        row["random_noise_gradient_rms_mT_per_m"]=noise_rms*1000
                        row["gradient_random_noise_snr_dB"]=20*math.log10(true_norm/noise_rms) if noise_rms else ""
                        summaries.append(row)
                print("Completed",domain,cname,model,flush=True)
    trial_handle.close()
    save_csv(out/"simulation_summary.csv",summaries)

    validations=dict(analytic_gradient_relative_fd_error=gradient_fd_error,
        all_three_source_distances_strictly_increase=True,
        minimum_ray_source_projection_m=np.min(projections),
        gradient_signal_decreases={},source_strength_constant_over_ray=True,
        synthetic_results_only=True)
    for model in moment_sets:
        ss=[r for r in summaries if r["domain"]=="ray" and r["configuration"]=="Config2" and r["moment_model"]==model and r["condition"]=="ideal"]
        vals=np.array([r["true_gradient_mean_mT_per_m"] for r in ss])
        validations["gradient_signal_decreases"][model]=bool(np.all(np.diff(vals)<0))
    source_codes=[Path(__file__),args.config,parameters_path,base/"simulation_helpers.py",
                  repo/"vendor/maps_estimator.py",repo/"vendor/mag_dipole_model.py"]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        code_hashes={str(p.relative_to(repo)):digest(p) for p in source_codes},
        config=cfg,source_positions_m=cs,source_axes=axes,sensor_coordinates_m=full_D,
        nominal_equal_moment_magnitude_A_m2=nominal_m,fitted_signed_moments_A_m2=fitted,
        ray_anchor_index=anchor["index"],ray_direction=normal,
        ray_first_mean_source_distance_mm=first_distance,
        distance_definition="arithmetic mean of the three source-center to array-center Euclidean distances",
        noise_is_assumed=True,residual_is_sensitivity_assumption=True,
        gradient_definition="mean Frobenius norm across three source gradients, in mT/m",
        snr_definition="20 log10(||X_true stacked||_F / RMS(||random-noise gradient stacked||_F))",
        validation=validations,parameter_provenance=parameters["provenance"],
        domain_note="The legacy domain name recorded denotes synthetic fields at fixed geometries derived from tracked poses. It contains no measured fields or experimental estimates.",
        summary_row_count=len(summaries),numerical_failure_count=sum(r["numerical_failures"] for r in summaries))
    save_json(out/"manifest.json",manifest)
    save_json(out/"validation.json",validations)
    print("DONE",json.dumps({"output":str(out),"rows":len(summaries),
        "nominal_moment":nominal_m,"fitted_moments":fitted.tolist(),
        "failures":manifest["numerical_failure_count"]}),flush=True)



linear_ops={}
if __name__=="__main__":
    main()
