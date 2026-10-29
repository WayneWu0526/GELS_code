#!/usr/bin/env python3
"""Export paper curves from full distance-sweep trial and summary tables."""
import argparse,csv,gzip,shutil
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh output directory: '+str(args.output))
    with (args.results/'simulation_summary.csv').open() as f:
        summaries=list(csv.DictReader(f))
    groups={}
    with gzip.open(args.results/'simulation_trials.csv.gz','rt') as f:
        for row in csv.DictReader(f):
            if row['domain']=='ray' and row['configuration']=='Config2' and row['condition']=='noise_and_residual':
                if row['success']!='1':
                    raise ValueError('Paper curves require all realizations to succeed')
                key=(row['moment_model'],int(row['point']))
                groups.setdefault(key,[]).append(float(row['gradient_relative_error']))
    args.output.mkdir(parents=True,exist_ok=True)
    for model in ['nominal20G','field_fit']:
        selected=sorted((r for r in summaries if r['domain']=='ray' and r['configuration']=='Config2' and r['condition']=='noise_and_residual' and r['moment_model']==model),key=lambda r:int(r['point']))
        rows=[]
        for r in selected:
            values=np.asarray(groups[(model,int(r['point']))])
            if len(values)!=int(r['trials']):raise ValueError('Trial count mismatch')
            rms=float(np.sqrt(np.mean(values**2)))
            rows.append(dict(moment_model=model,point=int(r['point']),distance_mm=float(r['distance_mm']),trials=len(values),
                position_error_mm_p10=float(r['position_error_mm_p10']),position_error_mm_median=float(r['position_error_mm_median']),position_error_mm_p90=float(r['position_error_mm_p90']),
                gradient_relative_total_error_rms=rms,gradient_signal_to_total_error_dB=-20*np.log10(rms),gradient_random_noise_snr_dB=float(r['gradient_random_noise_snr_dB'])))
        with (args.output/f'distance_extension_quality_{model}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copy2(Path(__file__).parent/'paper_figure/supp_distance_extension.tex',args.output/'supp_distance_extension.tex')
    print('Wrote paper figure tables to',args.output)
if __name__=='__main__':main()
