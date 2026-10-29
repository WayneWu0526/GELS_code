#!/usr/bin/env python3
"""Plot synthetic distance results and noise diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(p):
    return list(csv.DictReader(p.open()))


def values(rows, key):
    return np.array([float(r[key]) for r in rows])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results",type=Path)
    args=ap.parse_args()
    root=args.results
    out=root/"figures";out.mkdir(exist_ok=True)
    raw=read_csv(root/"simulation_summary.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":10,
        "axes.titlesize":11,"legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.8,
        "pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"})
    colors={"nominal20G":"#1769aa","field_fit":"#cf6827"}
    labels={"nominal20G":"20 G reference","field_fit":"Field-fitted amplitudes"}
    def select(domain,model,condition="noise_and_residual"):
        rr=[r for r in raw if r["domain"]==domain and r["configuration"]=="Config2" and r["moment_model"]==model and r["condition"]==condition]
        return sorted(rr,key=lambda r:float(r["distance_mm"]))
    def style(ax,xlabel=True):
        ax.grid(True,color="#e5e8eb",linewidth=.7)
        ax.set_axisbelow(True)
        if xlabel:ax.set_xlabel("Mean distance to three sources (mm)")
    def save(fig,name,pdf=True):
        fig.savefig(out/(name+".png"),dpi=220)
        fig.savefig(out/(name+".svg"))
        if pdf:fig.savefig(out/(name+".pdf"))
        plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(9.2,4.0))
    for model in colors:
        rr=select("ray",model);x=values(rr,"distance_mm")
        median=values(rr,"position_error_mm_median")
        axs[0].plot(x,median,color=colors[model],lw=2,label=labels[model])
        axs[0].fill_between(x,values(rr,"position_error_mm_p10"),values(rr,"position_error_mm_p90"),color=colors[model],alpha=.15,linewidth=0)
        axs[1].plot(x,values(rr,"true_gradient_mean_mT_per_m"),color=colors[model],lw=2,label=labels[model])
    axs[0].set_ylabel("Position error (mm)");axs[0].set_title("(a) Median error and 10–90% range",loc="left",fontweight="bold")
    axs[1].set_ylabel("True gradient norm (mT/m)");axs[1].set_title("(b) Mean norm of source gradients",loc="left",fontweight="bold")
    for ax in axs:
        ax.set_yscale("log");style(ax);ax.set_xlim(200,805)
    axs[0].legend(frameon=False,loc="lower right")
    fig.suptitle("Distance extension · every source distance increases · simulation only",fontsize=12,y=.97)
    fig.subplots_adjust(left=.08,right=.985,bottom=.23,top=.82,wspace=.30)
    fig.text(.08,.085,"Fixed source geometry, source moments and array orientation. Each position is stationary. Shaded bands contain 200 realizations.",fontsize=8,color="#525b66")
    fig.text(.08,.045,"0.3 µT noise per averaged slot axis + 6 µT fixed effective source-separated residual. These are stated assumptions, not measured noise.",fontsize=8,color="#525b66")
    save(fig,"distance_extension")

    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    ccolors={"ideal":"#6c7580","noise":"#208b80","noise_and_residual":"#1769aa"}
    clabels={"ideal":"Ideal measurements","noise":"Random noise only","noise_and_residual":"Noise + effective residual"}
    for condition in ccolors:
        rr=select("ray","nominal20G",condition);x=values(rr,"distance_mm")
        axs[0].plot(x,values(rr,"position_error_mm_median"),color=ccolors[condition],lw=1.7,label=clabels[condition])
        axs[2].plot(x,values(rr,"gradient_relative_error_median"),color=ccolors[condition],lw=1.7,label=clabels[condition])
    for model in colors:
        rr=select("ray",model,"noise")
        axs[1].plot(values(rr,"distance_mm"),values(rr,"gradient_random_noise_snr_dB"),color=colors[model],lw=1.8,label=labels[model])
    axs[0].set_yscale("log");axs[2].set_yscale("log")
    axs[1].axhline(0,color="#8c959e",ls=":",lw=1)
    axs[2].axhline(1,color="#8c959e",ls=":",lw=1)
    axs[0].set_ylabel("Position error (mm)");axs[1].set_ylabel("Gradient random-noise SNR (dB)");axs[2].set_ylabel("Relative total gradient error")
    for ax,title in zip(axs,["(a) Error components: 20 G","(b) Gradient random-noise SNR","(c) Gradient error: 20 G"]):
        style(ax);ax.set_title(title,loc="left",fontweight="bold");ax.set_xlim(200,805)
    axs[0].legend(frameon=False,fontsize=8,loc="lower right");axs[1].legend(frameon=False,fontsize=8)
    fig.subplots_adjust(left=.065,right=.985,bottom=.25,top=.87,wspace=.37)
    fig.text(.065,.10,"SNR uses random-noise gradient RMS only. Relative total gradient error also includes finite-array approximation and effective fixed residual.",fontsize=8,color="#525b66")
    fig.text(.065,.055,"Zero dB and relative error = 1 are diagnostic references, not universal loss-of-localization thresholds.",fontsize=8,color="#525b66")
    save(fig,"noise_and_gradient_diagnostics",pdf=False)

    print("Saved figures in",out)


if __name__=="__main__":main()
