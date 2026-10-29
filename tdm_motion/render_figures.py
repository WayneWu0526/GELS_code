"""Render reusable scientific figures with PGFPlots; no plotting Python packages."""
from pathlib import Path
import argparse
import csv
import numpy as np
import hashlib
import json
import re
import shutil
import subprocess

BASE = Path(__file__).resolve().parent

PREAMBLE = r'''\documentclass[tikz,border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\definecolor{cvblue}{RGB}{30,94,168}
\definecolor{ccorange}{RGB}{214,98,35}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.7cm},
width=8.7cm,height=6.8cm,scale only axis,
xmin=0,xmax=100,ymin=0,xlabel={Speed $v$ (mm/s)},
xtick={0,20,40,60,80,100},grid=major,
grid style={gray!20},axis line style={gray!65},
tick label style={font=\small},label style={font=\small},title style={font=\normalsize},
legend style={draw=none,font=\small,legend columns=3,/tikz/every even column/.append style={column sep=10pt}}]
'''


def plot_line(filename, metric, style, legend=None):
    result = rf'\addplot[{style}] table[x=speed_mm_s,y={metric},col sep=comma] {{../{filename}.csv}};'+'\n'
    if legend:
        result += r'\addlegendentry{'+legend+'}\n'
    return result


def render(results, kind):
    figdir = results/'figures'
    figdir.mkdir(exist_ok=True)
    text = PREAMBLE
    for column, metric, title, ylabel in [(1,'ep_mm','(a) Position error',r'$e_p$ (mm)'),
                                           (2,'er_deg','(b) Orientation error',r'$e_R$ (deg)')]:
        text += '\\nextgroupplot[title={'+title+'},ylabel={'+ylabel+'}'
        if column == 1:
            text += ',legend to name='+kind+'legend'
        text += ']\n'
        if kind == 'dynamic_errors':
            for mode, color in [('CV','cvblue'),('CC','ccorange')]:
                text += plot_line(mode+'_sequential_sensor',metric+'_mean',color+',thick,mark=*,mark size=1.7pt',mode+' mean' if column==1 else None)
            for mode, color in [('CV','cvblue'),('CC','ccorange')]:
                text += plot_line(mode+'_sequential_sensor',metric+'_p95',color+',thick,dashed',mode+' 95th percentile' if column==1 else None)
            text += plot_line('CV_synchronous',metric+'_mean','black!55,densely dotted,thick','Synchronous mean' if column==1 else None)
        else:
            for acquisition, style, name in [('sequential_sensor','solid,mark=*,mark size=1.5pt','all sensors sequential'),
                                              ('source_slot_only','dashed','sensors simultaneous per slot'),
                                              ('ideal_rho','densely dotted','ideal relative vectors')]:
                for mode, color in [('CV','cvblue'),('CC','ccorange')]:
                    text += plot_line(mode+'_'+acquisition,metric+'_mean',color+',thick,'+style,mode+': '+name if column==1 else None)
    text += '\\end{groupplot}\n'
    text += r'\node[anchor=south] at ($(group c1r1.north)!0.5!(group c2r1.north)+(0,0.65cm)$) {\pgfplotslegendfromname{'+kind+'legend}};\n'
    caption = ('12 sensors; radius 40 mm; fixed orientation; 180 matched reference phases. \\\\ '
               'Error reference: end of active-source acquisition. CV 3.47 Hz; CC 10.42 Hz.')
    if kind != 'dynamic_errors':
        caption = ('Means over the same 180 phases. Dashed curves remove within-slot sensor staggering. '
                   'Dotted curves also remove finite-array field/gradient approximation. Nominal timing model.')
    text += r'\node[anchor=north,align=center,text width=18cm,font=\small] at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.3cm)$) {'+caption+'};\n'
    text += '\\end{tikzpicture}\n\\end{document}\n'
    return compile_figure(figdir,kind,text)


def errorbar_tables(results):
    """Phase variability, not uncertainty of an estimated population mean."""
    with (results/'cases.csv').open() as f:
        cases = [r for r in csv.DictReader(f) if r['acquisition']=='sequential_sensor']
    with (results/'summary.csv').open() as f:
        summary = list(csv.DictReader(f))
    audit = {'statistic':'mean with 5th to 95th phase percentiles; NumPy linear quantiles',
             'interpretation':'deterministic variation across matched trajectory phases, not a confidence interval',
             'groups':[], 'plot_axes':{'x':'linear','y':'linear'}, 'plot_speed_range_mm_s':[0,10],
             'zero_speed':'Included in the main plot and retained in zero_speed_baseline.csv'}
    zero_rows=[]
    for mode in ['CV','CC']:
        rows=[]
        for speed in sorted({float(r['speed_mm_s']) for r in cases}):
            group=[r for r in cases if r['mode']==mode and float(r['speed_mm_s'])==speed]
            assert len(group)==180 and len({r['phase_index'] for r in group})==180
            row={'speed_mm_s':speed,'n':len(group)}
            original=next(r for r in summary if r['mode']==mode and r['acquisition']=='sequential_sensor' and float(r['speed_mm_s'])==speed)
            for metric in ['ep_mm','er_deg']:
                a=np.array([float(r[metric]) for r in group])
                mean=float(a.mean()); lo,hi=np.quantile(a,[.05,.95],method='linear')
                assert lo<=mean<=hi
                assert np.isclose(mean,float(original[metric+'_mean']),rtol=1e-12,atol=1e-12)
                assert np.isclose(hi,float(original[metric+'_p95']),rtol=1e-12,atol=1e-12)
                row.update({metric+'_mean':mean,metric+'_p05':lo,metric+'_p95':hi,
                            metric+'_minus':mean-lo,metric+'_plus':hi-mean})
            rows.append(row)
            audit['groups'].append({'mode':mode,'speed_mm_s':speed,'n':len(group)})
        with (results/'figures'/f'{mode}_errorbars.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        selected=[r for r in rows if 0<=r['speed_mm_s']<=10]
        assert [r['speed_mm_s'] for r in selected]==[0,1,2,5,10]
        with (results/'figures'/f'{mode}_errorbars_0_10.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
        zero_rows.extend(dict(mode=mode,**r) for r in rows if r['speed_mm_s']==0)
    with (results/'figures/zero_speed_baseline.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(zero_rows[0]));writer.writeheader();writer.writerows(zero_rows)
    (results/'figures/errorbar_validation.json').write_text(json.dumps(audit,indent=2)+'\n')


def scene_tex(geometry):
    """Zlab electromagnet geometry and styling, projected orthographically in TikZ."""
    center=np.array(geometry['center_m'])*1000
    ex=np.array([1,1,0])/np.sqrt(2)
    ey=np.array([-1,1,2])/np.sqrt(6)
    depth=np.cross(ex,ey)
    sources=np.array(geometry['source_positions_m'])*1000
    facets=[]; extents=[]; source_shapes=[]; outlines=[]
    theta=np.linspace(0,2*np.pi,65)
    for i,(p,m) in enumerate(zip(sources,geometry['source_moments_A_m2'])):
        axis=np.array(m)/np.linalg.norm(m)
        u=np.cross(axis,[0,0,1]);u/=np.linalg.norm(u);v=np.cross(axis,u)
        ring=np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v
        source_shape=[]
        for length,radius,color,alpha in [(250,20,'emcore',.42),(230,28,'emcoil',.34)]:
            a=p-axis*length/2+radius*ring;b=p+axis*length/2+radius*ring
            shape=np.vstack([a,b]);extents.extend(shape);source_shape.extend(shape)
            for j in range(64):
                poly=np.array([a[j],b[j],b[j+1],a[j+1]])
                shade=60+int(35*(1+np.cos(theta[j]-.6))/2)
                facets.append((poly.mean(axis=0)@depth,poly,f'{color}!{shade}!black',alpha))
        source_shapes.append(np.array(source_shape))
        outlines.append(([p-axis*125,p+axis*125],'black!88',.8))
        for offset in [-115,115]:
            outlines.append((p+axis*offset+28*ring,'emrim',.6))
    extent=np.array(extents)-center
    uv=np.column_stack([extent@ex,extent@ey])
    lower=uv.min(axis=0);upper=uv.max(axis=0)
    scale=min(8.25/(upper[0]-lower[0]),6.15/(upper[1]-lower[1]))
    middle=(lower+upper)/2
    def project(p):
        d=np.asarray(p)-center
        return np.array([4.65,4.75])+scale*(np.array([d@ex,d@ey])-middle)
    def coord(p):
        q=project(p);return f'({q[0]:.5f},{q[1]:.5f})'
    def path(points):return ' -- '.join(coord(p) for p in points)
    text=r'\node[anchor=west,font=\fontsize{10}{12}\selectfont] at (.3,8.5) {(a) Simulation scene};'+'\n'
    for p in sources:
        text+=r'\draw[gray!30,densely dashed,line width=.35pt] '+coord(p)+' -- '+coord(center)+';\n'
    for _,poly,fill,alpha in sorted(facets,key=lambda item:item[0]):
        text+=r'\path[fill='+fill+',fill opacity='+str(alpha)+',draw=none] '+path(poly)+' -- cycle;\n'
    for points,color,width in outlines:
        text+=r'\draw['+color+',opacity=.5,line width='+str(width)+'pt] '+path(points)+';\n'
    for i,(p,shape) in enumerate(zip(sources,source_shapes)):
        q=project(p);shape2=np.array([project(v) for v in shape])
        text+=f'\\node[font=\\fontsize{{10}}{{12}}\\selectfont,anchor=south] at ({q[0]:.3f},{shape2[:,1].max()+.12:.3f}) {{EM {i+1}}};\n'
    angles=np.linspace(0,2*np.pi,181)
    circle=center+geometry['radius_m']*1000*np.column_stack([np.cos(angles),np.sin(angles),np.zeros_like(angles)])
    text+=r'\draw[trajectory,line width=1.1pt] '+path(circle)+';\n'
    text+=r'\draw[trajectory,line width=1.1pt,->,>=stealth] '+path(circle[25:44])+';\n'
    target=circle[0]
    R=np.array(geometry['fixed_R']);offsets=np.array(geometry['sensor_offsets_m'])*1000
    # Sensor points are enlarged fivefold only for visibility; cylinder and circle share one scale.
    for off in offsets:
        text+=r'\fill[cvblue] '+coord(target+5*R@off)+' circle (.6pt);\n'
    tq=project(target)
    text+=r'\draw[gray!65,line width=.4pt] '+coord(target)+f' -- ({tq[0]+.8:.3f},{tq[1]-.7:.3f});\n'
    text+=f'\\node[anchor=north,font=\\fontsize{{9}}{{11}}\\selectfont] at ({tq[0]+.9:.3f},{tq[1]-.72:.3f}) {{12 sensors}};\n'
    cq=project(center)
    text+=f'\\node[anchor=north,text=trajectory!70!black,font=\\fontsize{{9}}{{11}}\\selectfont] at ({cq[0]-.6:.3f},{cq[1]-.8:.3f}) {{$r={geometry["radius_m"]*1000:g}\\,[\\mathrm{{mm}}]$}};\n'
    origin=np.array([.95,1.05])
    for axis,label in zip(np.eye(3),['x','y','z']):
        tip=origin+.75*np.array([axis@ex,axis@ey])
        text+=f'\\draw[->,>=stealth,gray!75,line width=.4pt] ({origin[0]},{origin[1]}) -- ({tip[0]:.3f},{tip[1]:.3f}) node[font=\\fontsize{{9}}{{11}}\\selectfont,anchor=south] {{$ {label} $}};\n'
    return text


def render_main(results):
    figdir=results/'figures';figdir.mkdir(exist_ok=True)
    errorbar_tables(results)
    text=r'''\documentclass[tikz,border=0pt,10pt]{standalone}
\usepackage{amsmath,pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{cvblue}{RGB}{30,94,168}
\definecolor{ccorange}{RGB}{214,98,35}
\definecolor{trajectory}{RGB}{28,143,111}
\definecolor{emcore}{HTML}{5A5A5A}
\definecolor{emcoil}{HTML}{B87333}
\definecolor{emrim}{HTML}{7A431B}
\begin{document}
\begin{tikzpicture}[every node/.style={font=\fontsize{9}{11}\selectfont}]
\path (0,0) -- (19.8,0);
'''
    text+=scene_tex(json.loads((results/'geometry.json').read_text()))
    for index,(metric,title,ylabel,y) in enumerate([
            ('ep_mm','(b) Position error',r'$e_p\,[\mathrm{mm}]$',4.9),
            ('er_deg','(c) Orientation error',r'$e_R\,[{}^\circ]$',.65)]):
        plotted=[]
        for mode in ['CV','CC']:
            with (figdir/f'{mode}_errorbars_0_10.csv').open() as f:
                plotted.extend(csv.DictReader(f))
        ymin=0.0
        ymax=max(float(r[metric+'_p95']) for r in plotted)*1.1
        text+=rf'''\begin{{axis}}[at={{(10.6cm,{y}cm)}},anchor=south west,width=8.85cm,height=3.05cm,scale only axis,
name=error{index},xmode=normal,ymode=normal,clip=false,
xmin=0,xmax=10,ymin={ymin},ymax={ymax},
title={{{title}}},title style={{at={{(0,1.09)}},anchor=south west,font=\fontsize{{10}}{{12}}\selectfont}},
ylabel={{{ylabel}}},xlabel={{{r'$v\,[\mathrm{mm/s}]$' if index else ''}}},
xtick={{0,2,4,6,8,10}},minor x tick num=0,{'xticklabels={},' if index==0 else ''}grid=major,grid style={{gray!15}},axis line style={{gray!60}},
tick label style={{font=\fontsize{{9}}{{11}}\selectfont}},label style={{font=\fontsize{{10}}{{12}}\selectfont}},
legend style={{at={{(0.03,.96)}},anchor=north west,draw=none,fill=white,font=\fontsize{{9}}{{11}}\selectfont,legend columns=2}}]
'''
        for mode,color,mark in [('CV','cvblue','*'),('CC','ccorange','square*')]:
            text+=rf'''\addplot[{color},line width=.8pt,mark={mark},mark size=1.5pt,
error bars/.cd,y dir=both,y explicit,error bar style={{line width=.5pt}},error mark options={{rotate=90,mark size=2pt,line width=.5pt}}]
table[x=speed_mm_s,y={metric}_mean,y error minus={metric}_minus,y error plus={metric}_plus,col sep=comma] {{{mode}_errorbars_0_10.csv}};
\addlegendentry{{{mode}}}
'''
        text+='\\end{axis}\n'
    text+='\\end{tikzpicture}\n\\end{document}\n'
    return compile_figure(figdir,'dynamic_errors',text)


def compile_figure(figdir,kind,text):
    source=figdir/(kind+'.tex');source.write_text(text)
    tex=shutil.which('pdflatex') or '/Library/TeX/texbin/pdflatex'
    with (figdir/(kind+'_build.log')).open('w') as log:
        for _ in range(2):
            subprocess.run([tex,'-interaction=nonstopmode','-halt-on-error',source.name],cwd=figdir,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    subprocess.run(['pdftoppm','-scale-to','2200','-singlefile','-png',str(figdir/(kind+'.pdf')),str(figdir/kind)],check=True,timeout=60)
    if kind == 'dynamic_errors':
        info=subprocess.check_output(['pdfinfo',str(figdir/(kind+'.pdf'))],text=True)
        width,height=map(float,re.search(r'Page size:\s+([\d.]+) x ([\d.]+)',info).groups())
        assert abs(width*2.54/72-19.8)<.001
        assert re.search(r'Pages:\s+1\n',info)
        geometry=json.loads((figdir.parent/'geometry.json').read_text())
        check=dict(pdf_width_cm=width*2.54/72,pdf_height_cm=height*2.54/72,
                   label_font_sizes_pt=[9,10],core_diameter_mm=40,core_length_mm=250,
                   coil_outer_diameter_mm=56,coil_length_mm=230,core_alpha=.42,coil_alpha=.34,line_alpha=.50,
                   trajectory_radius_mm=geometry['radius_m']*1000,pages=1,
                   x_axis='linear',y_axis='linear',plotted_speeds_mm_s=[0,1,2,5,10],
                   zero_speed_plotted=True)
        (figdir/'layout_validation.json').write_text(json.dumps(check,indent=2)+'\n')
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,figdir/(kind+'.pdf'),figdir/(kind+'.png')]}




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=BASE/'results')
    args=parser.parse_args()
    data={'dynamic_errors':render_main(args.results.resolve()),
          'sampling_diagnostics':render(args.results.resolve(),'sampling_diagnostics')}
    for name in ['CV_errorbars.csv','CC_errorbars.csv','CV_errorbars_0_10.csv','CC_errorbars_0_10.csv',
                 'zero_speed_baseline.csv','errorbar_validation.json','layout_validation.json']:
        p=args.results/'figures'/name
        data['dynamic_errors'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (args.results/'figures/manifest.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    main()
