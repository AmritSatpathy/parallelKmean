#!/usr/bin/env python3
"""Build the DSA5208 publication report without changing experimental inputs.

Run with Python containing python-docx, matplotlib, and PyMuPDF. Original
measurements and report.md remain independent source records. Only generated
publication artifacts are written by this builder.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from report_components import configure, figure, table, hyperlink

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report'
FIG = OUT / 'figures_publication'
BLUE, ORANGE, GREEN, GREY = '#003D7C', '#B65B12', '#22705A', '#626A73'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#171717', 'savefig.facecolor': 'white'})

INPUTS = ['scaling_big.csv', 'scaling_atlas.csv', 'scaling_atlas_2node.csv',
          'scaling_vanda_gpu.csv', 'scaling_vanda_multi.csv',
          'scaling_full_16cores_100iter.csv', 'scaling_vanda_multi_latest.csv']


def read(name):
    with (ROOT / 'results' / name).open() as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    fig.savefig(FIG / (name + '.png'), dpi=260, bbox_inches='tight', pad_inches=.10)
    fig.savefig(FIG / (name + '.svg'), bbox_inches='tight', pad_inches=.10)
    plt.close(fig)


def box(ax, x, y, w, h, title, sub='', color='#EAF1F5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',
                               facecolor=color, edgecolor='#546675', linewidth=.8))
    ax.text(x+w/2,y+h*.66 if sub else y+h/2,title,ha='center',va='center',
            fontsize=9,fontweight='bold',color=BLUE)
    if sub:
        ax.text(x+w/2,y+h*.29,sub,ha='center',va='center',fontsize=8,color='#263640')


def arrow(ax, a, b, dashed=False):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#3B4954',
                                           'lw':1,'linestyle':'--' if dashed else '-'})


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2,3.05)); ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    box(ax,.15,2.75,2.05,.9,'Rank 0: setup','CSV + k-means++')
    box(ax,3.05,2.75,3.0,.9,'Distribute once','Scatterv points; Bcast centroids')
    box(ax,7,2.75,2.8,.9,'Independent rank work','Assign points; sum + count')
    box(ax,7,.65,2.8,1.0,'Global reduction','3 × MPI_Allreduce',color='#EDF0F2')
    box(ax,3.05,.65,3,1,'Replicated update','Compute identical centroids')
    box(ax,.15,.65,2.05,1,'Final outputs','Inertia; labels; centroids',color='#E8F2EC')
    arrow(ax,(2.2,3.2),(3.05,3.2));arrow(ax,(6.05,3.2),(7,3.2))
    arrow(ax,(8.4,2.75),(8.4,1.65));arrow(ax,(7,1.15),(6.05,1.15))
    arrow(ax,(3.05,1.15),(2.2,1.15))
    ax.plot([4.5,4.5,8.4],[1.65,2.2,2.2],color=GREY,ls='--',lw=1)
    arrow(ax,(8.4,2.2),(8.4,2.75),True)
    ax.text(4.6,2.31,'Repeat until tolerance or iteration limit',fontsize=8,color=GREY)
    ax.text(.2,.12,'Solid arrows: data flow     Dashed arrows: iteration control',fontsize=8,color=GREY)
    save(fig,'01_mpi_architecture')

    fig,ax=plt.subplots(figsize=(7.2,3.05));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    for x,label in [(.2,'MPI rank 0 / GPU 0'),(5.25,'MPI rank 1 / GPU 1')]:
        ax.add_patch(FancyBboxPatch((x,.8),4.5,2.8,boxstyle='round,pad=0.02',
                                   facecolor='#FAFBFC',edgecolor='#A6B0B8',lw=.8))
        ax.text(x+.15,3.33,label,fontsize=9,color=BLUE,fontweight='bold')
        box(ax,x+.2,2.05,1.75,.85,'Host centroids','Updated every iteration')
        box(ax,x+2.25,2.05,2,.85,'CUDA assignment','1 thread / local point')
        box(ax,x+2.25,1,2,.7,'atomicAdd','Local sums / counts',color='#E8F2EC')
        box(ax,x+.2,1,1.75,.7,'Host partials','Copied from GPU')
        arrow(ax,(x+1.95,2.47),(x+2.25,2.47));arrow(ax,(x+3.25,2.05),(x+3.25,1.7))
        arrow(ax,(x+2.25,1.35),(x+1.95,1.35))
    ax.annotate('MPI_Allreduce on host buffers',xy=(6.05,.53),xytext=(3.85,.53),
                fontsize=8,ha='center',arrowprops={'arrowstyle':'<->','color':BLUE})
    ax.plot([1.27,1.27,6.32,6.32],[1,.3,.3,1],color=BLUE,lw=1)
    ax.text(.25,3.85,'Points remain on the GPU; centroids and partial reductions move each iteration.',fontsize=8,color=GREY)
    save(fig,'02_cuda_architecture')

    fig,axs=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
    for file,label,color,marker in [('scaling_big.csv','Local workstation',GREY,'s'),
            ('scaling_atlas.csv','Atlas: original 1-node series',BLUE,'o'),
            ('scaling_atlas_2node.csv','Atlas: original 2-node series',GREEN,'^')]:
        rows=read(file);base=float(next(r['time'] for r in rows if r['impl']=='serial'))
        rows=[r for r in rows if r['impl']=='mpi']; p=[int(r['nproc']) for r in rows]
        sp=[base/float(r['time']) for r in rows]
        axs[0].plot(p,sp,marker+'-',color=color,label=label,lw=1.3,ms=4)
        axs[1].plot(p,[s/n for s,n in zip(sp,p)],marker+'-',color=color,lw=1.3,ms=4)
    axs[0].plot([1,32],[1,32],'--',color='#999999',lw=.8,label='Ideal')
    axs[1].axhline(1,color='#999999',ls='--',lw=.8)
    for ax,t in zip(axs,['(a) Strong-scaling speedup','(b) Parallel efficiency']):
        ax.set_title(t,fontsize=9,loc='left');ax.set_xlabel('MPI ranks');ax.grid(alpha=.17)
        ax.set_xticks([1,8,16,32])
    axs[0].set_ylabel('Speedup / serial');axs[1].set_ylabel('Efficiency');axs[1].set_ylim(.45,1.08)
    axs[0].legend(fontsize=6.3,loc='upper left',frameon=False)
    save(fig,'03_original_cpu_scaling')

    fig,axs=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
    for file,label,c,m in [('scaling_vanda_multi.csv','Original collaborator series',BLUE,'o'),
                          ('scaling_vanda_multi_latest.csv','Ritik: Nov. 8 / jobs 1533324, 1533344',ORANGE,'s')]:
        rows=read(file);ts=[float(r['time']) for r in rows];ps=[int(r['nproc']) for r in rows]
        axs[0].plot(ps,ts,m+'-',color=c,label=label,ms=5,lw=1.3)
        axs[1].plot(ps,[ts[0]/v for v in ts],m+'-',color=c,ms=5,lw=1.3)
        for x,y in zip(ps,ts):axs[0].annotate(f'{y:.2f} s',(x,y),xytext=(6,3),textcoords='offset points',fontsize=8,color=c)
    axs[1].plot([1,2],[1,2],'--',color='#999999',lw=.8)
    for ax,t in zip(axs,['(a) GPU runtime','(b) Speedup within each series']):
        ax.set_title(t,fontsize=9,loc='left');ax.set_xlabel('A40 GPUs / MPI ranks');ax.set_xticks([1,2]);ax.grid(alpha=.17);ax.set_xlim(.9,2.35)
    axs[0].set_ylabel('Clustering time (s)');axs[1].set_ylabel('Speedup / 1 GPU')
    fig.legend(loc='outside lower center',fontsize=7,frameon=False,ncol=1)
    save(fig,'04_gpu_series')

    rows=read('scaling_full_16cores_100iter.csv');ts=[float(r['time']) for r in rows]
    fig,ax=plt.subplots(figsize=(7.2,1.65),layout='constrained')
    ax.barh(['MPI / 16 ranks','Serial / 1 core'],ts[::-1],color=[BLUE,GREY],height=.48)
    for y,t in enumerate(ts[::-1]):ax.text(t+25,y,f'{t:.2f} s',va='center',fontsize=9)
    ax.set_xlim(0,2020);ax.set_xlabel('Clustering time, 100 fixed iterations (s)');ax.grid(axis='x',alpha=.16)
    ax.set_axisbelow(True);save(fig,'05_atlas_full_load')

    fig,ax=plt.subplots(figsize=(7.2,2.15),layout='constrained')
    for filename,label,c in [('vanda_1gpu_full.OU','1 GPU / 1533324',BLUE),('vanda_2gpu_full.OU','2 GPUs / 1533344',ORANGE)]:
        text=(ROOT/'pbs/logs/vanda'/filename).read_text()
        vals=re.findall(r'iter\s+(\d+)\s+inertia\s+([0-9.e+]+)',text)
        ax.plot([int(i)+1 for i,_ in vals],[float(v)/1e9 for _,v in vals],color=c,label=label,lw=1.25)
    ax.set(xlabel='Iteration (1–32)',ylabel='Inertia (×10⁹)',xlim=(1,32));ax.grid(alpha=.17)
    ax.legend(fontsize=8,frameon=False);save(fig,'06_gpu_inertia')


def p(doc,text,style=None):
    return doc.add_paragraph(text,style)


def h(doc,text,level=1):
    return doc.add_heading(text,level)


def tbl(doc,num,title,headers,rows,widths):
    cap=p(doc,f'Table {num}. {title}', 'Table caption')
    cap.paragraph_format.keep_with_next=True
    return table(doc,headers,rows,widths,size=8.7)


def fig(doc,name,num,caption,width=6.40):
    figure(doc,FIG/(name+'.png'),f'Figure {num}. {caption}',width=width)


def code(doc,text):
    for line in text.splitlines():
        par=p(doc,line,'Code')
        par.paragraph_format.keep_with_next=False


def note(doc,label,text):
    par=p(doc,'','Note box');par.add_run(label+' ').bold=True;par.add_run(text)
    pr=par._p.get_or_add_pPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'F2F4F6');pr.append(sh)


def native_equation(doc,parts,number):
    par=p(doc,'');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    math=OxmlElement('m:oMath')
    for value,sub,sup in parts:
        container=math
        if sub is not None or sup is not None:
            tag='m:sSub' if sub is not None else 'm:sSup'
            expr=OxmlElement(tag);base=OxmlElement('m:e');expr.append(base);container=base
        r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=value;r.append(t);container.append(r)
        if sub is not None or sup is not None:
            arg=OxmlElement('m:sub' if sub is not None else 'm:sup')
            rr=OxmlElement('m:r');tt=OxmlElement('m:t');tt.text=sub if sub is not None else sup
            rr.append(tt);arg.append(rr);expr.append(arg);math.append(expr)
    par._p.append(math);par.add_run(f'     ({number})')


def configure_report(doc):
    configure(doc)
    doc.sections[0].header.paragraphs[0].runs[0].text='PARALLEL K-MEANS WITH MPI AND CUDA  |  DSA5208'
    for name,size,italic,bold in [('Code',8.5,False,False),('Table caption',9.4,False,False),
                                   ('Note box',9.4,False,False),('Reference',9,False,False)]:
        st=doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
        st.base_style=doc.styles['Normal'];st.font.name='Consolas' if name=='Code' else 'Times New Roman'
        st.font.size=Pt(size);st.font.italic=italic;st.font.bold=bold
        st.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT
        st.paragraph_format.space_after=Pt(4);st.paragraph_format.line_spacing=1.03
    doc.styles['Caption'].font.size=Pt(9.4)
    doc.styles['Caption'].paragraph_format.keep_with_next=False
    doc.styles['Reference'].paragraph_format.left_indent=Cm(.6)
    doc.styles['Reference'].paragraph_format.first_line_indent=Cm(-.6)
    doc.styles['Code'].paragraph_format.space_after=Pt(1)
    doc.core_properties.title='Parallel K-means clustering with MPI and CUDA'
    doc.core_properties.author='Amrit Satpathy; Ritik Srivastava'
    doc.core_properties.subject='DSA5208 scalable distributed computing: original and appended experiments'
    doc.core_properties.keywords='MPI, CUDA, K-means, Atlas, Vanda, provenance, NUS'


def build():
    before={name:digest(ROOT/'results'/name) for name in INPUTS}
    figures()
    doc=Document();configure_report(doc)
    authors=json.loads((ROOT/'assets/authors.json').read_text())
    # Compact cover: user-requested density overrides the template's large quiet area.
    par=p(doc,'');par.paragraph_format.space_after=Pt(10)
    border=OxmlElement('w:pBdr');line=OxmlElement('w:bottom')
    for k,v in [('val','single'),('sz','8'),('color','003D7C')]:line.set(qn('w:'+k),v)
    border.append(line);par._p.get_or_add_pPr().append(border)
    par=p(doc,'Parallel K-means clustering with MPI and CUDA','Title');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par=p(doc,'Distributed centroid reduction and GPU acceleration on Atlas and Vanda','Subtitle');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par=p(doc,'');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(ROOT/'assets/nus-logo-orange-b-stack.tmb-ze-139-182.png'),width=Cm(3.5))
    par=p(doc,'DSA5208 · Scalable Distributed Computing for Data Science');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par.runs[0].bold=True
    table(doc,['AUTHOR','STUDENT ID'],[(a['name'],a['student_id']) for a in sorted(authors,key=lambda a:a['name'])],[10.3,6.1],size=9.9)
    par=p(doc,'National University of Singapore · November 2026');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par=p(doc,'Source code and reproducible experiment records');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par=p(doc,'');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    hyperlink(par,'github.com/AmritSatpathy/parallelKmean','https://github.com/AmritSatpathy/parallelKmean')
    h(doc,'Project scope',2)
    p(doc,'A C++ serial reference, a distributed MPI implementation, and an MPI+CUDA implementation of Lloyd’s algorithm. The study examines point decomposition, global centroid reductions, numerical agreement, and strong scaling on NUS HPC systems.')
    h(doc,'Evidence retained in this edition',2)
    p(doc,'The original collaborator experiments remain intact. Ritik Srivastava’s Atlas job 1034257.venus01 and Vanda jobs 1533323, 1533324, and 1533344 are appended as separate runs, with their own workload definitions and identifiers. No original measurement or scheduler identifier is replaced.')
    note(doc,'Principal results.','The original Atlas series reaches 31.32× speedup at 32 MPI ranks. The appended Atlas run achieves 15.74× at 16 ranks over 100 iterations. The appended Vanda runs scale from 10.159337 s on one A40 to 5.073951 s on two A40s.')
    doc.add_page_break()
    h(doc,'Contents')
    toc=p(doc,'');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'TOC \\o "1-1" \\h \\z \\u')
    rr=OxmlElement('w:r');tt=OxmlElement('w:t');tt.text='Contents are updated automatically when rendered.';rr.append(tt);field.append(rr);toc._p.append(field)
    p(doc,'The main text presents the algorithm, implementation choices, validation, and results. Appendix A records scheduler identifiers; Appendix B gives reproducibility commands and artifact provenance. Original and appended experiment series are explicitly distinguished throughout.')
    h(doc,'Reading the results',2)
    p(doc,'All runtime tables report clustering-loop time unless explicitly labelled PBS wall time. Each speedup uses a baseline from the same experiment series. A 32-iteration GPU run is not compared directly with the separate 100-iteration Atlas run.')
    note(doc,'Provenance convention.','“Original series” denotes the experiments already included in the collaborator’s report and CSV files. “Appended series” denotes Ritik’s newly supplied logs and CSV files. A missing historical job ID is stated as unavailable; no identifier is inferred from a runtime or filename.')
    doc.add_page_break()
    h(doc,'Abstract')
    p(doc,'K-means combines independent nearest-centroid searches with a globally coupled centroid update, making it a useful case study in distributed and accelerator computing. We implement Lloyd’s algorithm in serial C++, CPU MPI, and MPI+CUDA, sharing the same data representation and k-means++ seeding. MPI partitions points across ranks and combines local centroid statistics through three Allreduce collectives per iteration. CUDA replaces each rank’s local assignment loop with one thread per point while keeping cross-rank reductions on host buffers. Experiments use MNIST for real-data interpretation and synthetic Gaussian blobs for strong scaling. The retained original Atlas experiments achieve 31.32× speedup at 32 ranks, with 97.88% efficiency; the original Vanda series approximately halves runtime from one to two A40 GPUs. An independent appended Atlas experiment executes 100 fixed iterations over three million 64-dimensional points, reducing clustering time from 1703.059729 s to 108.220590 s at 16 ranks. Appended Vanda experiments execute 32 fixed iterations in 10.159337 s on one A40 and 5.073951 s on two, giving 2.002× within-series speedup. Matching displayed inertia supports numerical agreement within each series. The analysis distinguishes timed iteration work from initialization and scheduler wall time, retains both authors’ experiment records, and limits conclusions where dataset hashes, repeated timings, or historical scheduler IDs were not retained.')
    h(doc,'1. Introduction and objectives')
    p(doc,'The project investigates how decomposition and synchronization affect the parallel scaling of a nontrivial algorithm. Lloyd’s K-means alternates nearest-centroid assignment and centroid recomputation [1]. The assignment step is independent across points, but the update requires statistics from every partition. The complete algorithm therefore requires communication and coordinated iteration, satisfying the assignment’s requirement to go beyond an embarrassingly parallel workload.')
    p(doc,'Our objectives are to establish a serial correctness reference, explain the MPI design and its CUDA extension, measure strong scaling on Atlas and Vanda, and interpret clustering quality on MNIST. The implementation remains within the MPI+CUDA scope. Larger algorithms or additional programming frameworks are unnecessary to answer these questions.')
    p(doc,'Two sets of experiments contribute to the report. The original collaborator series provides CPU scaling through 32 ranks and GPU scaling through two devices. Ritik’s appended series supplies an additional 100-iteration Atlas workload and independently logged one- and two-GPU Vanda runs. Both sets are preserved as scientific evidence rather than collapsed into a single best-time table.')
    h(doc,'2. Algorithm and datasets')
    h(doc,'2.1 Objective, initialization, and iteration',2)
    p(doc,'For n points xᵢ in d dimensions, K centroids μⱼ, and cluster assignment a(i), K-means minimizes the within-cluster sum of squared distances, or inertia. The squared distance avoids a square root without changing the nearest-centroid decision.')
    native_equation(doc,[('J = ∑', 'i=1…n',None),(' ‖x', 'i',None),(' − μ','a(i)',None),('‖',None,'2')],1)
    p(doc,'All variants use k-means++ initialization [2], with a fixed seed for comparisons within a series. Each iteration assigns points to their nearest centroid, accumulates sums and counts, and replaces each nonempty centroid by the arithmetic mean of its assigned points. Empty clusters retain their previous centroid. Assignment costs O(nKd); centroid accumulation costs O(nd). K-means++ seeding costs O(nKd) overall and is performed once outside the timed clustering loop.')
    p(doc,'The default convergence path tests relative inertia change; the CPU implementations also test whether assignments stopped moving. Benchmark runs instead pass --fixed-iter so that all configurations execute the same number of iterations. The log phrase “converged in” is therefore interpreted as “completed” for these fixed-iteration measurements, without claiming that a convergence criterion was met.')
    h(doc,'2.2 Dataset definitions and comparability',2)
    tbl(doc,1,'Data used for quality analysis and performance experiments.',
        ['Dataset / series','Points','Features','K','Iterations'],[
        ['MNIST, original quality study','70,000','784','10','Convergence'],
        ['Synthetic, original CPU/GPU','3,000,000','64','128','32 fixed'],
        ['Synthetic, appended Atlas','3,000,000','64','128','100 fixed'],
        ['Synthetic, appended Vanda','3,000,000','64','128','32 fixed'],
        ['MNIST, Vanda smoke test','70,000','784','10','10 fixed']], [6.7,2.4,2.3,1.6,3.4])
    p(doc,'MNIST combines 70,000 digit images, each flattened into 784 raw pixel features; column 0 is excluded from features and retained as a ground-truth digit label [3]. Synthetic Gaussian blobs contain planted cluster labels, again in column 0. The generator supplies controllable n, d, K, seed, and separation, making it suitable for a fixed-workload scaling experiment.')
    note(doc,'Separate workloads.','The appended Atlas inertia is 1.214576×10⁹, whereas the original and appended Vanda synthetic runs report approximately 8.624057×10⁸. Matching shape and filename do not prove identical data. Dataset bytes, generator settings, or initialization may differ; without hashes, these are separate workloads. Their times are never pooled into a controlled cross-platform speedup.')
    h(doc,'3. Parallel architecture and design decisions')
    h(doc,'3.1 MPI decomposition and global synchronization',2)
    p(doc,'Figure 1 shows the distributed execution path. Rank 0 reads the numeric CSV and seeds centroids. MPI_Scatterv sends near-equal contiguous point blocks to ranks; the first n mod p ranks receive one extra point. MPI_Bcast distributes the common initial centroids. A point belongs to exactly one partition, while all ranks keep the small K×d centroid array.')
    fig(doc,'01_mpi_architecture',1,'Point decomposition and the globally coupled update. Solid arrows show data flow; dashed arrows show iteration control.')
    p(doc,'Every rank performs a local assignment loop and produces K×d coordinate sums, K counts, and scalar diagnostics. Three MPI_Allreduce calls combine these arrays, leaving every rank with the global statistics needed to calculate identical new centroids. This follows the collective semantics defined by MPI [4]. The coordinate-sum reduction communicates 8192 floats for K=128 and d=64, independent of n; counts and diagnostics add small separate messages.')
    p(doc,'Allreduce is used instead of a root-only Reduce followed by a separate broadcast because every rank immediately needs the updated global state. Keeping coordinates and counts in separate collectives preserves their different MPI datatypes. Counts use long long; inertia uses double. CPU sums and input points use float, conserving memory but allowing small changes from floating-point reduction order.')
    h(doc,'3.2 MPI+CUDA execution and memory movement',2)
    p(doc,'CUDA changes how each rank computes its local statistics. Figure 2 shows the implemented host-staged pipeline. Each rank selects a GPU, copies local points once, and launches a kernel with 256 threads per block and one thread per point. Centroids use shared memory when the K×d array fits; otherwise the kernel reads them from global memory. Threads accumulate sums, counts, and inertia with atomicAdd. CUDA execution and memory behavior follow the toolkit programming guide [5].')
    fig(doc,'02_cuda_architecture',2,'Two-rank GPU path. Partial statistics return to host memory before MPI_Allreduce; this implementation does not perform reductions directly on device pointers.')
    p(doc,'Before each kernel launch, current centroids are copied to the device and partial accumulators are zeroed. After the kernel completes, small partial arrays are copied to the host. MPI combines them, each rank updates centroids on the host, and the cycle repeats. This avoids transferring the large point array every iteration, although it still incurs host/device synchronization and atomic contention.')
    p(doc,'GPU selection is rank modulo visible device count. This supports the measured one-node, one- and two-GPU jobs when the requested devices are visible to each rank. A general multi-node extension should use node-local rank and verify CUDA_VISIBLE_DEVICES. No claim of multi-node GPU validation is made here.')
    h(doc,'3.3 Trade-offs and expected scaling limits',2)
    p(doc,'Contiguous near-equal partitions balance a uniform O(Kd) assignment cost per point. Replicating centroids avoids remote reads during nearest-centroid searches. The global reductions provide the necessary dependency between iterations, so removing them would change Lloyd’s algorithm. At high rank counts, fixed collective latency and shrinking local work can reduce efficiency. CPU memory bandwidth, SIMD behavior, cache reuse, and GPU atomic contention may also matter; the saved timings do not isolate these mechanisms.')
    p(doc,'Rank 0 remains responsible for CSV parsing and initialization and retains the full input while local blocks are allocated. This limits end-to-end scalability and memory distribution, even when the clustering loop scales well. Optional label gathering and output also occur after the reported clustering interval. These boundaries are kept explicit throughout the evaluation.')
    h(doc,'4. Experimental protocol and timing')
    p(doc,'Strong scaling holds the dataset, K, seed, and iteration budget fixed while increasing ranks or GPUs. We calculate speedup and efficiency using the serial baseline for CPU MPI, and the one-GPU baseline for GPU-only scaling. The use of different baselines is intentional and labelled in every table.')
    native_equation(doc,[('S(p) = T', 'serial',None),(' / T(p),     E(p) = S(p) / p',None,None)],2)
    p(doc,'For GPU scaling, Tserial in Equation (2) is replaced by the one-GPU runtime and p denotes GPU count. A value slightly above 100% efficiency is retained as measured, rather than capped; it may reflect ordinary timing variation and does not establish superlinear behavior. Each comparison is made inside its own experiment series.')
    tbl(doc,2,'Execution environments, with original and appended settings retained separately.',
        ['Series','Platform / configuration','Toolchain / evidence'],[
        ['Original local','Ryzen 7 6800HS; 8 cores / 16 threads; RTX 3050 Ti','GCC 13; OpenMPI 4.1; original report'],
        ['Original Atlas','1-node 16-rank and 2-node 32-rank series','GCC 13.2; OpenMPI 4.1.6; original report'],
        ['Original Vanda','1–2 NVIDIA A40 GPUs','GCC 13.2; OpenMPI 4.1.6; CUDA 12.4'],
        ['Appended Atlas','cnode-33-43-27; 16 ranks / 1 node','GCC 7.3 / OpenMPI 4.1.1 from session; saved log'],
        ['Appended Vanda','GN-A40-021 and GN-A40-035; 1–2 A40s','GCC 13.2; OpenMPI 4.1.6; CUDA 12.4 in logs']],
        [3.2,6.5,6.7])
    p(doc,'The CPU harness runs serial before the MPI configurations. REPS controls repetition and retains the minimum successful time, while fixed iterations ensure comparable iteration counts. The appended Atlas and Vanda full runs have one recorded timing per configuration. Historical repetition counts are not inferred when the retained CSV does not encode them. The source uses MPI barriers around the distributed timed loop; the GPU loop also synchronizes CUDA work before completing an iteration.')
    p(doc,'Reported time is the clustering loop, including per-iteration computation, host/device copies where applicable, and collective communication. PBS wall time additionally includes build, data generation, CSV read, initialization, process launch, and other job overhead. These are different performance measures. For the appended Vanda logs, CSV loading alone takes 24.840 s on one GPU and 23.348 s on two, outside the reported 10.159337 s and 5.073951 s clustering times.')
    h(doc,'5. Correctness and real-data interpretation')
    h(doc,'5.1 Numerical agreement within each experiment series',2)
    tbl(doc,3,'Validation evidence. Original reported tests and newly retained logs are distinguished.',
        ['Evidence','Comparison','Observed result'],[
        ['Original MNIST validation','Serial versus 8-rank MPI','100% label agreement; inertia 1.784862×10¹¹; 13 iterations'],
        ['Original small blobs','Serial versus 2/4/8-rank MPI','99.7–99.9% label agreement; inertia difference ~10⁻⁶'],
        ['Original large blobs','Serial / MPI / MPI+CUDA','Inertia 8.624057–8.624058×10⁸'],
        ['Appended Atlas, 100 iterations','Serial versus 16 ranks','Both print inertia 1.214576×10⁹'],
        ['Appended Vanda, 32 iterations','1 versus 2 GPUs','Both print inertia 8.624057×10⁸'],
        ['Appended GPU MNIST smoke test','1 GPU, 10 iterations','Inertia 1.785707×10¹¹; exit status 0 (session record)']],
        [4.1,4.9,7.4])
    p(doc,'The original label comparisons and purity tests are retained from the collaborator’s report; their original label files and raw validation transcripts are not present in this checkout. New inertia values are read from the appended CSV and log records. Equality at printed precision supports numerical agreement but does not prove bitwise-equal centroids or labels. Floating-point addition is non-associative, so MPI reduction order and CUDA atomic order can shift boundary assignments [4], [5].')
    p(doc,'Figure 3 uses the newly retained per-iteration Vanda log lines. Both GPU trajectories drop rapidly and then approach the same displayed objective. This provides evidence that parallel execution preserves the iterative behavior, in addition to matching the final diagnostic. The inertia is evaluated during assignment before the subsequent centroid update, matching the source implementation.')
    fig(doc,'06_gpu_inertia',3,'Recorded objective trajectories for Vanda jobs 1533324 and 1533344. The nearly overlapping curves show within-series numerical agreement.')
    h(doc,'5.2 Conclusions about MNIST',2)
    p(doc,'The original study reports approximately 0.585 cluster purity for K=10 on raw MNIST pixels. Purity is the sum of each predicted cluster’s largest true-label count divided by n. Reported majority fractions are approximately 0.94 for a 0-dominated cluster, 0.91 for a 2-dominated cluster, and 0.86 for a 6-dominated cluster. Stroke-similar digits such as 4/7/9 and 3/5/8 are confused. These observations are clustering results, not supervised classification accuracy.')
    p(doc,'Raw-pixel Euclidean distance captures broad shape but is sensitive to position and handwriting variation. Ten digit labels do not force an unsupervised method to discover ten perfectly corresponding groups. The original synthetic quality study reports 0.977 purity with 126 of 128 clusters perfectly pure; this planted-label result is retained as originally reported. Feature transformations or alternative clustering methods are outside this project’s MPI+CUDA scope.')
    h(doc,'6. Results: original collaborator experiments')
    p(doc,'This section retains the earlier CPU and GPU experiments. Their CSV files remain unchanged. Historical scheduler identifiers were not encoded in those files or the original report, so they are not replaced by the newer jobs. Original results and appended results are different evidence series even where workload dimensions match.')
    h(doc,'6.1 CPU strong scaling on the workstation and Atlas',2)
    a=read('scaling_atlas.csv');b=read('scaling_atlas_2node.csv');local=read('scaling_big.csv')
    def by_rank(rows):return {int(r['nproc']):r for r in rows if r['impl']=='mpi'}
    aa,bb,ll=map(by_rank,[a,b,local])
    basea=float(a[0]['time']);baseb=float(b[0]['time']);basel=float(local[0]['time'])
    rows=[]
    for n in [1,2,4,8,16,32]:
        rows.append([str(n),f"{float(ll[n]['time']):.6f}" if n in ll else '—',
                     f"{float(aa[n]['time']):.6f}" if n in aa else '—',
                     f"{float(bb[n]['time']):.6f}",f"{baseb/float(bb[n]['time']):.3f}",
                     f"{100*baseb/float(bb[n]['time'])/n:.2f}%"])
    tbl(doc,4,'Original MPI clustering runtimes and two-node-series scaling (32 fixed iterations).',
        ['Ranks','Local (s)','Atlas 1-node (s)','Atlas 2-node (s)','Speedup¹','Efficiency¹'],rows,
        [1.3,3.0,3.3,3.4,2.6,2.8])
    p(doc,f'¹ Two-node-series metrics use its own serial baseline of {baseb:.6f} s. Separate serial baselines are {basel:.6f} s locally and {basea:.6f} s for the single-node Atlas series. The 1-rank row above is MPI, not serial. The two-node allocation was described in the original report as filling one node through 16 ranks, then using both at 32; per-rank placement logs were not retained.')
    fig(doc,'03_original_cpu_scaling',4,'Original strong scaling. Curves retain separate serial baselines; the workstation shows a stronger efficiency decline at 16 ranks.')
    p(doc,'The original two-node Atlas series reaches 31.3216× speedup at 32 ranks and 97.88% efficiency. The separate single-node series reaches 15.7335× at 16 ranks. Locally, speedup is 6.8116× at eight ranks and 9.1431× at 16 ranks. The workstation’s eight physical cores and 16 logical threads make shared execution resources a plausible contributor to the local efficiency decline; the timings do not separately measure SMT, bandwidth, or communication costs.')
    h(doc,'6.2 Original CUDA results and cross-platform context',2)
    old=read('scaling_vanda_multi.csv');t1,t2=[float(r['time']) for r in old]
    tbl(doc,5,'Original GPU measurements retained from the collaborator’s report and CSV files.',
        ['Configuration / source','Time (s)','Within-series comparison'],[
        ['Local RTX 3050 Ti (original report)','35.2','~16.98× versus local serial 597.873896 s'],
        ['Vanda 1 A40, standalone CSV','8.627849','Independent earlier 1-GPU measurement'],
        ['Vanda 1 A40, paired-series CSV',f'{t1:.6f}','GPU baseline'],
        ['Vanda 2 A40s, paired-series CSV',f'{t2:.6f}',f'{t1/t2:.4f}× speedup; {100*t1/t2/2:.2f}% efficiency']],
        [7.1,3.1,6.2])
    p(doc,'The original report compared the best two-GPU time of 4.309527 s with its Atlas serial baseline of 404.566169 s, giving approximately 93.88×. This comparison is retained as the original cross-platform reference, not as a processor-only speedup or the baseline for the appended runs. Hardware, compiler, and placement differ. The local 35.2 s GPU time is available in the original report rather than a retained standalone result log.')
    h(doc,'7. Results: appended Atlas and Vanda experiments')
    h(doc,'7.1 Atlas full-load validation: 100 fixed iterations',2)
    ar=read('scaling_full_16cores_100iter.csv');s,m=[float(r['time']) for r in ar];sp=s/m
    tbl(doc,6,'Appended Atlas job 1034257.venus01, run by Ritik Srivastava on 5 November 2026.',
        ['Implementation','Ranks','Time (s)','Speedup','Efficiency'],[
        ['Serial','1',f'{s:.6f}','1.0000','100.00%'],
        ['MPI CPU','16',f'{m:.6f}',f'{sp:.4f}',f'{100*sp/16:.2f}%']],
        [5.6,1.7,3.5,2.8,2.8])
    fig(doc,'05_atlas_full_load',5,'Appended Atlas clustering-loop runtime. Both configurations process the same workload for 100 fixed iterations.')
    p(doc,'The saved log identifies cnode-33-43-27 and reports exit status 0, PBS wall time 00:32:56, and aggregate CPU time 01:09:50. Peak host memory is 1,707,136 KB. Serial and 16-rank MPI both print inertia 1.214576×10⁹. The speedup is 15.7369× and efficiency is 98.36%, consistent with effective point partitioning at this workload size. This is a separate two-configuration check rather than a replacement for the original scaling curve.')
    h(doc,'7.2 Independent Vanda one- and two-GPU runs',2)
    vr=read('scaling_vanda_multi_latest.csv');v1,v2=[float(r['time']) for r in vr];vsp=v1/v2
    tbl(doc,7,'Appended Vanda experiments by Ritik Srivastava on 8 November 2026; 32 fixed iterations.',
        ['PBS job ID','Node','GPUs / ranks','Time (s)','GPU speedup'],[
        ['1533324.stdct-mgmt-02','GN-A40-021','1 / 1',f'{v1:.6f}','1.0000'],
        ['1533344.stdct-mgmt-02','GN-A40-035','2 / 2',f'{v2:.6f}',f'{vsp:.4f}']],
        [5.4,3.4,2.6,2.8,2.2])
    fig(doc,'04_gpu_series',6,'Original and appended Vanda series are both retained. Speedup uses each series’ own one-GPU baseline; no best-time replacement is performed.')
    p(doc,f'The appended GPU speedup is {vsp:.4f}×, with {100*vsp/2:.2f}% efficiency. This is approximately ideal scaling; one observation per configuration cannot establish statistically significant superlinear performance. Both runs print final inertia 8.624057×10⁸ and the terminal session records exit status 0. The supplied log copies contain complete iteration trajectories but omit PBS footer summaries; scheduler wall times of 00:01:43 and 00:00:53 are therefore attributed to the recorded qstat output, not inferred from the shorter log files.')
    p(doc,'The logs confirm GCC 13.2.0, OpenMPI 4.1.6, CUDA 12.4, and compilation for sm_86. They also contain nonfatal OpenMPI warnings that libpbs.so.0 could not be loaded for the TM launcher and resource components; those messages were explicitly ignored and the one-node runs produced results. Successful local execution does not establish functioning multi-node PBS integration. Host memory figures reported by PBS are not GPU VRAM utilization.')
    h(doc,'8. Discussion, limitations, and conclusion')
    h(doc,'8.1 What the measurements establish',2)
    p(doc,'The original Atlas curves and appended full-load run both show that local assignment work can dominate collective overhead at the tested problem size. The small centroid statistics explain why communication can be amortized over millions of points. The one-to-two-GPU results in both Vanda series support the same decomposition across two accelerators. These measurements support the design, while a communication-only profile would be needed to attribute exact percentages of runtime to Allreduce, computation, or memory movement.')
    p(doc,'K-means is not embarrassingly parallel as a whole: every centroid update depends on globally reduced statistics and precedes the next assignment. The CUDA implementation keeps that dependency intact while increasing local throughput. Near-linear measured speedup is evidence for these tested configurations, not a guarantee that efficiency remains high at 96 CPU ranks or across multiple GPU nodes.')
    h(doc,'8.2 Limits of evidence and interpretation',2)
    p(doc,'Only strong scaling is measured; weak scaling is an optional extension. GPU experiments cover at most two A40s on a single node. Original data files do not carry dataset hashes, per-run build manifests, or complete historical scheduler IDs. New runs provide stronger job-level provenance, but dataset identity across platforms remains unverified. Similar printed inertia is a useful diagnostic rather than a complete proof of identical cluster assignments.')
    p(doc,'The appended runs contain one retained timing per configuration and no confidence intervals. The original harness selects a minimum when repeated, but the raw repetitions are not stored in the CSV. Comparisons should therefore be interpreted as measured examples under shared-cluster conditions. Scheduler allocation counts may exceed active MPI ranks: Vanda assigned 36 CPU slots for one GPU and 72 for two in the session record, while the code launched only one or two ranks.')
    p(doc,'The original report described larger Atlas requests and four-GPU requests as limited by resource availability. Those observations remain part of the original study; this edition adds no unexecuted benchmark. The observed queue message “Insufficient amount of resource: ncpus” or “node_pool” identifies a scheduling constraint at that moment, not a permanent capacity or queue-policy limit.')
    h(doc,'8.3 Conclusion',2)
    p(doc,'The project delivers serial, MPI CPU, and MPI+CUDA implementations of the same Lloyd iteration, with explicit global centroid reductions and real HPC timing evidence. Original Atlas scaling reaches 31.32× at 32 ranks; the independently appended Atlas run achieves 15.74× at 16 ranks over 100 iterations. The appended Vanda pair achieves approximately 2.00× speedup from one to two A40s, matching the qualitative result of the retained original GPU series. MNIST illustrates the limits of raw-pixel clustering, while numerical comparisons support correctness within each series. The principal contribution is the measured relationship between point-level parallel work and the synchronized centroid update, with reproducibility boundaries and contributor provenance stated explicitly.')
    h(doc,'9. AI assistance acknowledgment and author responsibility')
    p(doc,'The authors acknowledge use of generative AI in this project. The tools reported by the team are GPT 6.1 Astra, GPT 6.1 Solo, GPT 6 Astra, Claude, and Kiro (retained from the original declaration). The GPT and Claude names are recorded as supplied by the authors; this declaration does not assert a verified vendor release identifier or a particular Claude version. ChatGPT/Codex assisted with repository review, C++/MPI/CUDA code and script drafting or debugging, HPC compiler and library troubleshooting, PBS submission and monitoring commands, result interpretation, report editing, reproducible charts, architecture diagrams, and document layout. Kiro’s earlier assistance with scaffolding, code, documentation, and local experiments remains acknowledged.')
    p(doc,'Both authors retain responsibility for the submitted project, including the implementation, experiment execution, dataset handling, correctness checks, numerical claims, citations, and final text. AI assistance does not substitute for the authors’ understanding or transfer responsibility to a tool. Empirical claims in this edition are linked to retained CSV files, supplied execution logs, the original report, or explicitly identified session records. AI tools are not treated as independent sources of experimental validation.')
    p(doc,'NUS’s Policy for the Use of AI in Teaching and Learning, current version 15 September 2026, requires acknowledgment of AI use and treats unacknowledged AI-generated work as an academic-integrity issue; assignment-specific guidance also applies [6]. The project brief explicitly requests an explanation of AI usage. NUS Libraries provides citation guidance and notes IEEE’s acknowledgment practice for AI-generated content [7]. This section discloses the tools and the assisted tasks; the acknowledgment record is also included in the numbered references [8].')
    h(doc,'References')
    refs=[
        ('S. Lloyd, “Least squares quantization in PCM,” IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 129–137, 1982.','https://doi.org/10.1109/TIT.1982.1056489'),
        ('D. Arthur and S. Vassilvitskii, “k-means++: The advantages of careful seeding,” Proc. 18th ACM–SIAM Symposium on Discrete Algorithms, pp. 1027–1035, 2007.','https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf'),
        ('Y. LeCun, C. Cortes, and C. J. C. Burges, “The MNIST database of handwritten digits.” Dataset source.','https://yann.lecun.com/exdb/mnist/'),
        ('Message Passing Interface Forum, MPI: A Message-Passing Interface Standard, version 4.1, November 2023. Standard reference; deployed OpenMPI versions are separately recorded in Table 2.','https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf'),
        ('NVIDIA, CUDA C++ Programming Guide, toolkit version 12.4, 2024.','https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html'),
        ('National University of Singapore, University Policy Workgroup for AI in Teaching and Learning, Policy for the Use of AI in Teaching and Learning, current version 15 September 2026, §§2.5–2.6, 3–4. Accessed 8 November 2026.','https://ctlt.nus.edu.sg/wp-content/uploads/2026/04/Policy-for-Use-of-AI-in-Teaching-and-Learning.pdf'),
        ('NUS Libraries, “Citation Styles: Citing AI Tools,” updated 28 August 2026. Accessed 8 November 2026.','https://libguides.nus.edu.sg/citation/ai'),
        ('A. Satpathy and R. Srivastava, “AI assistance acknowledgment for Parallel K-means,” unpublished project disclosure, November 2026. GPT 6.1 Astra, GPT 6.1 Solo, GPT 6 Astra, Claude, ChatGPT/Codex, and Kiro; roles and author responsibility declared in §9.',None),
        ('A. Satpathy and R. Srivastava, Parallel K-means project repository: source, original report, experiment CSV files, and appended HPC logs, 2026. Referenced throughout Appendices A–B.','https://github.com/AmritSatpathy/parallelKmean')]
    for i,(text,url) in enumerate(refs,1):
        par=p(doc,f'[{i}] {text} ','Reference')
        if url:hyperlink(par,'Online source',url)
    doc.add_page_break()
    h(doc,'Appendix A. Experiment provenance and scheduler ledger')
    p(doc,'The ledger preserves the original collaborator series and appends Ritik’s experiments. Source artifacts are cited as project records [9]. Historical job IDs absent from the supplied files remain unavailable rather than being assigned newer identifiers. Local artifact checksums are recorded in report/experiment_manifest.json; these checksums identify the retained evidence files, not the original dataset bytes.')
    tbl(doc,'A1','Original experiment files preserved without replacing measurements or assigning new job IDs.',
        ['Original artifact in results/','Coverage','Scheduler ID'],[
        ['scaling_big.csv','Local serial + MPI ranks 1–16','Not recorded / local'],
        ['scaling_atlas.csv','Atlas serial + MPI ranks 1–16','Not retained in supplied CSV/report'],
        ['scaling_atlas_2node.csv','Atlas serial + MPI ranks 1–32','Not retained in supplied CSV/report'],
        ['scaling_vanda_gpu.csv','Original standalone 1-GPU run','Not retained in supplied CSV/report'],
        ['scaling_vanda_multi.csv','Original paired 1- and 2-GPU runs','Not retained in supplied CSV/report']],
        [6.4,4.7,5.3])
    tbl(doc,'A2','Successful appended experiments. Full scheduler names are retained where available.',
        ['Job identifier','Purpose / outcome','Evidence retained'],[
        ['1027948.venus01','Atlas MNIST sweep; exit 0; 00:04:33','Terminal history; no downloaded log'],
        ['1034257.venus01','Atlas synthetic serial + 16 ranks; 100 iterations; exit 0','pbs/logs/1034257.venus01.OU'],
        ['1533323.stdct-mgmt-02','Vanda MNIST smoke test; 10 iterations; exit 0','Terminal history; log not downloaded'],
        ['1533324.stdct-mgmt-02','Vanda 1 A40; 32 iterations; exit 0','pbs/logs/vanda/vanda_1gpu_full.OU'],
        ['1533344.stdct-mgmt-02','Vanda 2 A40s; 32 iterations; exit 0','pbs/logs/vanda/vanda_2gpu_full.OU']],
        [5.1,6.0,5.3])
    p(doc,'The Vanda log copies were renamed descriptively during download. Their full PBS identifiers are preserved from the terminal’s original log filenames: 1533324.stdct-mgmt-02.OU and 1533344.stdct-mgmt-02.OU. The log bodies corroborate node, ranks, toolkit, iteration count, inertia, and runtime; they do not themselves contain a JobId field.')
    tbl(doc,'A3','Earlier Atlas setup attempts and queued requests retained from the session history.',
        ['Job identifier','Recorded state / issue'],[
        ['1027931.venus01','Exit 1; data/blobs_big.csv could not be opened before data/ existed'],
        ['1027936.venus01','Exit 1; obsolete /app1/ebenv referenced missing rec_modules.sh'],
        ['1027943.venus01','Exit 1; GCC/OpenMPI modulefiles unavailable in batch environment'],
        ['1027945.venus01','Exit 1; modulefile discovery still failed'],
        ['1027946.venus01','Exit 2; inherited MVAPICH2 library caused linker errors'],
        ['1027967.venus01','96-core request queued; cancellation instructed; final state not retained']],
        [5.2,11.2])
    p(doc,'Failed setup jobs and queued requests are not included as runtime benchmarks. Their identifiers document the environment troubleshooting that led to a clean GCC/OpenMPI build. The two-GPU job 1533344 had initially queued for node_pool resources and completed before the attempted qdel; its recorded successful result is therefore retained.')
    h(doc,'Appendix B. Reproducibility, architecture details, and artifacts')
    h(doc,'B.1 Source-to-experiment map',2)
    tbl(doc,'B1','Implementation and experiment entry points.',
        ['File','Role'],[
        ['src/common.hpp','Numeric CSV, k-means++, distance, timing, output'],
        ['src/kmeans_serial.cpp','Serial correctness and performance reference'],
        ['src/kmeans_mpi.cpp','Scatterv decomposition and global centroid statistics'],
        ['src/kmeans_mpi_cuda.cu','GPU assignment and host-staged MPI reductions'],
        ['scripts/run_scaling.sh','CPU sweep; fixed iterations and optional best-of-repetitions'],
        ['pbs/atlas_sweep.pbs; pbs/vanda_cuda.pbs','PBS CPU/GPU execution, build, and dataset generation']],
        [7.4,9.0])
    p(doc,'A benchmark run uses --label-col 0, the selected seed, --max-iter N, and --fixed-iter. MNIST validation runs may omit --fixed-iter to test convergence and export labels. The commands below illustrate reproduction using the provided source; they are not additional execution claims.')
    h(doc,'B.2 Atlas and Vanda submission examples',2)
    code(doc,'mkdir -p bin data results pbs/logs\nmake serial mpi gen\nqsub -l select=1:ncpus=16:mpiprocs=16 \\\n+  -l walltime=02:00:00 \\\n+  -v "CSV=data/blobs_big.csv,K=128,PROCS=16,MAXIT=100,REPS=1" \\\n+  pbs/atlas_sweep.pbs')
    p(doc,'On Atlas 9 the working script uses explicit GCC 7.3 and OpenMPI 4.1.1 paths and a clean library path. Its settings should not be replaced with an unrelated Atlas module setup. Compilation uses the compute node’s CPU flags, including -march=native; binaries are rebuilt on the target system.')
    code(doc,'module load GCC/13.2.0 OpenMPI/4.1.6-GCC-13.2.0 CUDA/12.4.0\nqsub -l select=1:ncpus=2:ngpus=2:mpiprocs=2 \\\n+  -l walltime=01:00:00 \\\n+  -v "NP=2,CSV=data/blobs_big.csv,K=128,MAXIT=32,SEED=42,CUDA_ARCH=sm_86" \\\n+  pbs/vanda_cuda.pbs')
    p(doc,'For one GPU set ncpus=1, ngpus=1, mpiprocs=1, and NP=1. Both GPU counts should use the same dataset file and seed. PBS directives must not have trailing inline comments. Submit from the repository root and pre-create pbs/logs and data; compilation and large data generation run inside the allocation.')
    h(doc,'B.3 Validation and job inspection',2)
    code(doc,'python3 scripts/compare_labels.py results/serial.labels results/mpi.labels\npython3 scripts/validate.py --pred results/mpi.labels \\\n+  --csv data/mnist.csv --truth-col 0\nqstat -x -f JOB_ID\ngrep "^RESULT" pbs/logs/JOB_ID*.OU')
    p(doc,'A successful PBS exit code is checked together with a RESULT line, expected n/d/K, rank count, iteration count, and final inertia. The CPU harness suppresses subprocess stderr and may skip failed repetitions, so a final exit code alone is insufficient to establish that all expected rows exist. The appended Atlas CSV contains both required rows and the Vanda log copies contain the expected results.')
    h(doc,'B.4 Data preservation and report build',2)
    p(doc,'The report builder reads CSV files and logs but does not modify them or submit jobs. Generated charts are stored under report/figures_publication/ as PNG and SVG. report/experiment_manifest.json records artifact hashes and known run IDs. The editable report is report/Parallel_Kmeans_Project_Report.docx, and its rendered counterpart is report/Parallel_Kmeans_Project_Report.pdf. The original report/report.md is retained with an appended experiment addendum.')
    p(doc,'Use distinct dataset hashes and experiment filenames in future runs. The original and appended series must remain distinguishable when generating charts; the 100-iteration Atlas values must not replace a 32-iteration baseline. Archived original measurements without saved scheduler IDs remain valid supplied records, with that provenance limit stated explicitly.')
    # Explicit table geometry is re-applied after every row exists.
    for t in doc.tables:
        widths=[int(c.get(qn('w:w'))) for c in t._tbl.tblGrid]
        for row in t.rows:
            for cell,w in zip(row.cells,widths):
                cell._tc.get_or_add_tcPr().find(qn('w:tcW')).set(qn('w:w'),str(w))
        pr=t._tbl.tblPr;indent=OxmlElement('w:tblInd');indent.set(qn('w:w'),'80');indent.set(qn('w:type'),'dxa');pr.append(indent)
    dest=OUT/'Parallel_Kmeans_Project_Report.docx';doc.save(dest)
    after={name:digest(ROOT/'results'/name) for name in INPUTS}
    assert before==after,'Experimental inputs changed unexpectedly'
    records=[]
    for file in INPUTS:
        records.append({'path':'results/'+file,'sha256':before[file],
                        'series':'appended' if file in INPUTS[-2:] else 'original', 'rows':read(file)})
    logs=['pbs/logs/1034257.venus01.OU','pbs/logs/vanda/vanda_1gpu_full.OU','pbs/logs/vanda/vanda_2gpu_full.OU']
    manifest={'title':'Parallel K-means report evidence','report_date':'2026-11-08',
              'authors':authors,'original_inputs_preserved':True,'experiments':records,
              'logs':[{'path':f,'sha256':digest(ROOT/f)} for f in logs],
              'appended_run_ids':['1027948.venus01','1034257.venus01','1533323.stdct-mgmt-02','1533324.stdct-mgmt-02','1533344.stdct-mgmt-02'],
              'original_run_ids':'Not recorded in original supplied report/CSV; no replacement IDs assigned',
              'interpretation':'No cross-series CPU/GPU speedup; dataset hashes are unavailable.'}
    (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest)
    print('Original result files unchanged; 6 figures generated; provenance manifest written.')


if __name__=='__main__':
    build()
