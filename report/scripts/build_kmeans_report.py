#!/usr/bin/env python3
"""Build the DSA5208 publication report without changing experimental inputs.

Run with Python containing python-docx, matplotlib, and PyMuPDF. Original
measurements remain independent source records. Markdown and publication
artifacts are generated from the same report content.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from statistics import median
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph
from docx.table import Table

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
          'scaling_full_16cores_100iter.csv', 'scaling_vanda_multi_latest.csv',
          'scaling_32cores_100iter_1036751.csv', 'scaling_48cores_100iter_1036776.csv',
          'scaling_12cores_100iter_1036952.csv', 'scaling_16cores_100iter_1037018.csv',
          'scaling_24cores_100iter_1037036.csv']

ATLAS_RUNS = [
    {'job_id':'1036952.venus01','ranks':12,'node':'cnode-33-43-26',
     'csv':'scaling_12cores_100iter_1036952.csv','log':'pbs/logs/atlas/1036952.venus01.OU',
     'walltime':'00:35:22'},
    {'job_id':'1034257.venus01','ranks':16,'node':'cnode-33-43-27',
     'csv':'scaling_full_16cores_100iter.csv','log':'pbs/logs/1034257.venus01.OU',
     'walltime':'00:32:56'},
    {'job_id':'1037018.venus01','ranks':16,'node':'cnode-33-43-53',
     'csv':'scaling_16cores_100iter_1037018.csv','log':'pbs/logs/atlas/1037018.venus01.OU',
     'walltime':'00:32:02'},
    {'job_id':'1037036.venus01','ranks':24,'node':'cnode-33-43-53',
     'csv':'scaling_24cores_100iter_1037036.csv','log':'pbs/logs/atlas/1037036.venus01.OU',
     'walltime':'00:31:30'},
    {'job_id':'1036751.venus01','ranks':32,'node':'cnode-33-43-21',
     'csv':'scaling_32cores_100iter_1036751.csv','log':'pbs/logs/atlas/1036751.venus01.OU',
     'walltime':'00:36:17'},
    {'job_id':'1036776.venus01','ranks':48,'node':'cnode-33-43-47',
     'csv':'scaling_48cores_100iter_1036776.csv','log':'pbs/logs/atlas/1036776.venus01.OU',
     'walltime':'00:31:13'}]


def read(name):
    with (ROOT / 'results' / name).open() as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atlas_metrics():
    records=[]
    for run in ATLAS_RUNS:
        rows=read(run['csv']);serial=next(r for r in rows if r['impl']=='serial')
        mpi=next(r for r in rows if r['impl']=='mpi')
        assert int(mpi['nproc'])==run['ranks']
        assert all(r['iters']=='100' and r['n']=='3000000' and r['dims']=='64' and r['k']=='128' for r in rows)
        assert serial['inertia']==mpi['inertia']=='1.214576e+09'
        log=(ROOT/run['log']).read_text()
        assert re.search(r'Exit Status:\s*0\b',log) and run['job_id'] in log
        assert serial['time'] in log and mpi['time'] in log
        cpu_time=re.search(r'CPU Time Used:\s*(\S+)',log)
        memory=re.search(r'Memory Used:\s*(\S+)',log)
        st,mt=float(serial['time']),float(mpi['time'])
        records.append({**run,'serial_time':st,'mpi_time':mt,'speedup':st/mt,
                        'efficiency':st/mt/run['ranks'],'exit_status':0,
                        'pbs_footer_cpu_time':cpu_time.group(1) if cpu_time else None,
                        'pbs_footer_memory':memory.group(1) if memory else None})
    return records


def save(fig, name):
    fig.savefig(FIG / (name + '.png'), dpi=260, bbox_inches='tight', pad_inches=.10)
    fig.savefig(FIG / (name + '.svg'), bbox_inches='tight', pad_inches=.10)
    plt.close(fig)


CARD_TEXT=[]


def icon(ax,kind,x,y,s=.45,color=BLUE):
    """Small vector hardware/operation symbols; no raster brand imitation."""
    lw=1.05
    if kind in ('cpu','gpu'):
        ax.add_patch(FancyBboxPatch((x+s*.16,y+s*.15),s*.68,s*.7,
            boxstyle='round,pad=0.01,rounding_size=0.025',fill=False,edgecolor=color,lw=lw))
        for a in [.27,.5,.73]:
            for side in [0,1]:
                ax.plot([x+s*(.04 if side==0 else .84),x+s*(.16 if side==0 else .96)],
                        [y+s*a,y+s*a],color=color,lw=.9)
                ax.plot([x+s*a,x+s*a],[y+s*(.03 if side==0 else .85),y+s*(.15 if side==0 else .97)],color=color,lw=.9)
        for a in [.3,.53]:
            for b in [.32,.57]:
                ax.add_patch(Rectangle((x+s*a,y+s*b),s*.16,s*.16,facecolor=color,lw=0))
    elif kind=='server':
        for a in [.16,.53]:
            ax.add_patch(FancyBboxPatch((x+s*.08,y+s*a),s*.84,s*.26,
                boxstyle='round,pad=0.01,rounding_size=0.02',fill=False,edgecolor=color,lw=lw))
            ax.add_patch(Circle((x+s*.24,y+s*(a+.13)),s*.035,color=color))
            ax.plot([x+s*.45,x+s*.78],[y+s*(a+.13)]*2,color=color,lw=.8)
    elif kind=='network':
        for dx,dy in [(.5,.75),(.2,.25),(.8,.25)]:
            ax.plot([x+s*.5,x+s*dx],[y+s*.5,y+s*dy],color=color,lw=lw)
            ax.add_patch(Circle((x+s*dx,y+s*dy),s*.11,facecolor='white',edgecolor=color,lw=lw))
    elif kind=='database':
        ax.add_patch(Rectangle((x+s*.18,y+s*.18),s*.64,s*.58,facecolor='white',edgecolor=color,lw=lw))
        for yy in [.18,.47,.76]:
            ax.add_patch(Ellipse((x+s*.5,y+s*yy),s*.64,s*.22,facecolor='white' if yy==.76 else 'none',edgecolor=color,lw=lw))
    elif kind=='reduce':
        for dx in [.18,.5,.82]:
            ax.add_patch(Circle((x+s*dx,y+s*.79),s*.07,facecolor=color,lw=0))
            ax.plot([x+s*dx,x+s*.5],[y+s*.7,y+s*.32],color=color,lw=lw)
        ax.add_patch(Circle((x+s*.5,y+s*.23),s*.12,facecolor=color,lw=0))
    elif kind=='refresh':
        ax.add_patch(Circle((x+s*.5,y+s*.5),s*.3,fill=False,edgecolor=color,lw=lw))
        ax.annotate('',xy=(x+s*.78,y+s*.66),xytext=(x+s*.89,y+s*.4),
                    arrowprops={'arrowstyle':'-|>','color':color,'lw':lw})
    elif kind=='output':
        ax.add_patch(Rectangle((x+s*.18,y+s*.12),s*.64,s*.76,fill=False,edgecolor=color,lw=lw))
        ax.plot([x+s*.3,x+s*.46,x+s*.72],[y+s*.48,y+s*.3,y+s*.67],color=color,lw=1.5)
    else:
        for yy in [.24,.5,.76]:ax.plot([x+s*.2,x+s*.8],[y+s*yy]*2,color=color,lw=lw)


def card(ax,x,y,w,h,title,body,kind='cpu',step=None,color='#EDF3F9'):
    """Cards reserve explicit icon, title, body and edge-padding regions."""
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',
                               facecolor=color,edgecolor='#BBC9D6',linewidth=.8))
    icon(ax,kind,x+.18,y+h-.61,s=.43)
    title_artist=ax.text(x+.79,y+h-.36,title,ha='left',va='center',
                         fontsize=8.7,fontweight='bold',color=BLUE)
    body_artist=ax.text(x+.24,y+.28,body,ha='left',va='bottom',fontsize=8.0,
                        color='#354758',linespacing=1.40)
    CARD_TEXT.extend([(ax,title_artist,(x+.76,y+h-.63,w-.98,.51)),
                      (ax,body_artist,(x+.21,y+.21,w-.42,h-.66))])


def card_qa(fig):
    """Shrink only within a readable range; fail on any unsafe text geometry."""
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax,artist,(x,y,w,h) in CARD_TEXT:
        if artist.figure is not fig:
            continue
        lo=ax.transData.transform((x,y));hi=ax.transData.transform((x+w,y+h))
        extent=artist.get_window_extent(renderer)
        while (extent.x0<lo[0] or extent.y0<lo[1] or extent.x1>hi[0] or extent.y1>hi[1]) and artist.get_fontsize()>6.8:
            artist.set_fontsize(artist.get_fontsize()-.2)
            fig.canvas.draw();extent=artist.get_window_extent(renderer)
        if extent.x0<lo[0]-1 or extent.y0<lo[1]-1 or extent.x1>hi[0]+1 or extent.y1>hi[1]+1:
            raise ValueError(f'Architecture card text exceeds safe padding: {artist.get_text()} '
                             f'box={extent.bounds} safe=({lo},{hi}) size={artist.get_fontsize()}')


def arrow(ax, a, b, dashed=False):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#3B4954',
                                           'lw':1,'linestyle':'--' if dashed else '-'})


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2,3.35)); ax.set(xlim=(0,12),ylim=(0,5.4));ax.axis('off')
    icon(ax,'server',.13,4.75,.48)
    ax.text(.85,5.02,'NUS HPC  /  distributed MPI execution',fontsize=10,fontweight='bold',color=BLUE,va='center')
    ax.text(.85,4.71,'Atlas: validated through 48 ranks · partitioned points',fontsize=8,color=GREY)
    xs=[.2,4.38,8.56];w=3.24;hh=1.45
    card(ax,xs[0],2.97,w,hh,'Read and initialize','Rank 0 reads numeric CSV\nShared k-means++ seed',kind='database',step=1)
    card(ax,xs[1],2.97,w,hh,'Distribute points','Scatterv point partitions\nBcast initial centroids',kind='network',step=2)
    card(ax,xs[2],2.97,w,hh,'Assign locally','Find nearest centroid\nLocal sums + counts',kind='cpu',step=3)
    card(ax,xs[2],.68,w,hh,'Reduce globally','3 × MPI_Allreduce\nGlobal sums + counts',kind='reduce',step=4,color='#F0F2F5')
    card(ax,xs[1],.68,w,hh,'Update centroids','Global sum / count\nReplicate new centroids',kind='refresh',step=5)
    card(ax,xs[0],.68,w,hh,'Write outputs','Final inertia\nLabels + centroids',kind='output',step=6,color='#EAF3EF')
    for a,b in [(0,1),(1,2)]:arrow(ax,(xs[a]+w,3.61),(xs[b],3.61))
    arrow(ax,(10.18,2.97),(10.18,2.13));arrow(ax,(xs[2],1.32),(xs[1]+w,1.32));arrow(ax,(xs[1],1.32),(xs[0]+w,1.32))
    ax.plot([6,6,9.12,9.12],[2.13,2.48,2.48,2.97],color=ORANGE,ls=(0,(3,3)),lw=1)
    arrow(ax,(9.12,2.65),(9.12,2.97),True)
    ax.text(6,2.65,'Next iteration',fontsize=7.5,color=ORANGE)
    ax.text(.25,.12,'DATA FLOW  →     ITERATION CONTROL  - - →',fontsize=7,color=GREY)
    card_qa(fig)
    save(fig,'01_mpi_architecture')

    fig,ax=plt.subplots(figsize=(7.2,3.90));ax.set(xlim=(0,12),ylim=(0,6.4));ax.axis('off')
    icon(ax,'server',.15,5.84,.48)
    ax.text(.85,6.14,'NUS HPC  /  MPI + CUDA on Vanda',fontsize=10,fontweight='bold',color=BLUE,va='center')
    ax.text(.85,5.82,'One MPI rank per visible A40 GPU · point arrays uploaded once',fontsize=8,color=GREY)
    for x,label in [(.2,'MPI rank 0  /  A40 GPU 0'),(6.35,'MPI rank 1  /  A40 GPU 1')]:
        ax.add_patch(FancyBboxPatch((x,1.18),5.45,4.27,boxstyle='round,pad=0.025,rounding_size=0.10',
                                   facecolor='#F9FAFC',edgecolor='#AABBCB',lw=.8))
        icon(ax,'gpu',x+.19,4.82,.45,color=GREEN)
        ax.text(x+.82,5.08,label,fontsize=8.5,color=BLUE,fontweight='bold',va='center')
        # Two full-width stages per rank keep text readable and well inside cards.
        card(ax,x+.25,3.05,4.94,1.45,'CUDA assignment kernel','One thread per point; centroids cached\nDevice atomicAdd → sums, counts, inertia',kind='gpu',color='#EAF3EF')
        card(ax,x+.25,1.42,4.94,1.45,'Host partial statistics','Copy device partials to host buffers\nPrepare sums + counts + diagnostics',kind='cpu')
        arrow(ax,(x+2.72,3.05),(x+2.72,2.87))
    # A dedicated reduction lane avoids crossing any card or text label.
    ax.plot([2.92,2.92,9.07,9.07],[1.42,.94,.94,1.42],color=BLUE,lw=1)
    ax.add_patch(FancyBboxPatch((3.3,.18),5.4,.5,boxstyle='round,pad=.02,rounding_size=.06',
                               facecolor='#EDF0F4',edgecolor='#BCC8D4',lw=.8))
    ax.text(6,.43,'MPI_Allreduce on host buffers',ha='center',va='center',fontsize=8.3,color=BLUE,fontweight='bold')
    arrow(ax,(6,.94),(6,.68))
    ax.text(.25,.19,'SOLID: STATISTICS FLOW',fontsize=6.8,color=GREY)
    ax.text(9,.19,'HOST-STAGED REDUCTIONS',fontsize=6.3,color=GREY,ha='left')
    card_qa(fig)
    save(fig,'02_cuda_architecture')

    fig,axs=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
    for file,label,color,marker in [('scaling_big.csv','Local workstation',GREY,'s'),
            ('scaling_atlas.csv','Atlas: 1-node allocation',BLUE,'o'),
            ('scaling_atlas_2node.csv','Atlas: 2-node allocation',GREEN,'^')]:
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
    for file,label,c,m in [('scaling_vanda_multi.csv','Vanda measurement batch A',BLUE,'o'),
                          ('scaling_vanda_multi_latest.csv','Vanda measurement batch B · jobs 1533324 / 1533344',ORANGE,'s')]:
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

    runs=atlas_metrics();ps=sorted({r['ranks'] for r in runs})
    grouped={p:[r for r in runs if r['ranks']==p] for p in ps}
    med=lambda key:[median(r[key] for r in grouped[p]) for p in ps]
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.75),layout='constrained')
    axs[0].plot(ps,med('serial_time'),'s--',color=GREY,label='Job-matched serial',ms=4)
    axs[0].plot(ps,med('mpi_time'),'-',color=BLUE,label='MPI',lw=1.3)
    axs[0].scatter([r['ranks'] for r in runs],[r['mpi_time'] for r in runs],color=BLUE,s=18,zorder=3)
    for rank,t in zip(ps,med('mpi_time')):
        axs[0].annotate(f'{t:.2f} s',(rank,t),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7,color=BLUE)
    axs[0].set_yscale('log');axs[0].set_ylabel('Clustering time (s, log scale)')
    axs[0].set_title('(a) 100-iteration runtime',fontsize=9,loc='left');axs[0].legend(fontsize=7,frameon=False,loc='center right')
    axs[1].plot(ps,med('speedup'),'-',color=BLUE,label='Measured',lw=1.3)
    axs[1].scatter([r['ranks'] for r in runs],[r['speedup'] for r in runs],color=BLUE,s=18,zorder=3)
    axs[1].plot([0,48],[0,48],'--',color=GREY,label='Ideal',lw=.8)
    axs[1].set_title('(b) Speedup / matched serial',fontsize=9,loc='left');axs[1].set_ylabel('Speedup')
    axs[1].legend(fontsize=7,frameon=False,loc='upper left');axs[1].set_ylim(0,52)
    for ax in axs:
        ax.set_xlabel('MPI ranks');ax.set_xticks(ps);ax.set_xlim(9,52);ax.grid(alpha=.17)
    save(fig,'05_atlas_full_load')

    fig,ax=plt.subplots(figsize=(7.2,2.15),layout='constrained')
    for filename,label,c,style,marker,offset in [
            ('vanda_1gpu_full.OU','1 GPU / 1533324',BLUE,'-','o',0),
            ('vanda_2gpu_full.OU','2 GPUs / 1533344',ORANGE,(0,(4,3)),'D',2)]:
        text=(ROOT/'pbs/logs/vanda'/filename).read_text()
        vals=re.findall(r'iter\s+(\d+)\s+inertia\s+([0-9.e+]+)',text)
        ax.plot([int(i)+1 for i,_ in vals],[float(v)/1e9 for _,v in vals],
                color=c,label=label,lw=1.8 if offset==0 else 1.3,linestyle=style,
                marker=marker,markevery=(offset,4),markersize=4.3,
                markerfacecolor='white',markeredgewidth=1.1)
    ax.set(xlabel='Iteration (1–32)',ylabel='Inertia (×10⁹)',xlim=(1,32));ax.grid(alpha=.17)
    ax.legend(fontsize=8,frameon=False,handlelength=3.2);save(fig,'06_gpu_inertia')


def p(doc,text,style=None):
    return doc.add_paragraph(text,style)


def h(doc,text,level=1):
    par=doc.add_heading(text,level)
    if level==1:
        idx=sum(1 for x in doc.paragraphs if x.style.name=='Heading 1')
        mark=OxmlElement('w:bookmarkStart');mark.set(qn('w:id'),str(idx));mark.set(qn('w:name'),f'section_{idx}')
        end=OxmlElement('w:bookmarkEnd');end.set(qn('w:id'),str(idx))
        par._p.insert(0,mark);par._p.append(end)
    return par


def tbl(doc,num,title,headers,rows,widths):
    cap=p(doc,f'Table {num}. {title}', 'Table caption')
    cap.paragraph_format.keep_with_next=True
    return table(doc,headers,rows,widths,size=8.7)


def fig(doc,name,num,caption,width=6.40):
    figure(doc,FIG/(name+'.png'),f'Figure {num}. {caption}',width=width)


def code(doc,text):
    for line in text.replace('\n+','\n').splitlines():
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
    doc.styles['Caption'].font.bold=False
    doc.styles['Caption'].paragraph_format.keep_with_next=False
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption',
                 'Table caption','Note box','Reference','Header','Footer','Code']:
        st=doc.styles[name];rf=st.element.get_or_add_rPr().get_or_add_rFonts()
        for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
            rf.attrib.pop(qn('w:'+key),None)
        for key in ['ascii','hAnsi','eastAsia','cs']:
            rf.set(qn('w:'+key),'Consolas' if name=='Code' else 'Times New Roman')
    doc.styles['Reference'].paragraph_format.left_indent=Cm(.6)
    doc.styles['Reference'].paragraph_format.first_line_indent=Cm(-.6)
    doc.styles['Code'].paragraph_format.space_after=Pt(1)
    doc.core_properties.title='Parallel K-means clustering with MPI and CUDA'
    doc.core_properties.author='Amrit Satpathy; Ritik Srivastava'
    doc.core_properties.subject='DSA5208 scalable distributed computing: unified project experiments'
    doc.core_properties.keywords='MPI, CUDA, K-means, Atlas, Vanda, provenance, NUS'


def write_markdown(doc):
    """Keep the readable Markdown report synchronized with the designed edition."""
    candidates=[ROOT/'assets/nus_logo_full-vertical.jpg',*FIG.glob('*.png')]
    image_paths={digest(path):path.relative_to(OUT).as_posix() if path.is_relative_to(OUT)
                 else '../'+path.relative_to(ROOT).as_posix() for path in candidates}
    lines=['<!-- Generated from the completed-run publication report. -->','']
    in_code=False
    for element in doc.element.body:
        if element.tag==qn('w:p'):
            par=Paragraph(element,doc);text=par.text.strip();style=par.style.name
            if style!='Code' and in_code:
                lines.extend(['```','']);in_code=False
            for blip in element.findall('.//'+qn('a:blip')):
                rid=blip.get(qn('r:embed'));part=doc.part.related_parts[rid]
                path=image_paths.get(hashlib.sha256(part.blob).hexdigest())
                if path:lines.extend([f'![Project figure]({path})',''])
            math=element.find('.//'+qn('m:oMath'))
            if math is not None:
                value=''.join(n.text or '' for n in math.iter(qn('m:t')))
                lines.extend([value+' '+text,'']);continue
            # Hyperlink text is not included in python-docx Paragraph.text.
            for link in element.findall(qn('w:hyperlink')):
                label=''.join(n.text or '' for n in link.iter(qn('w:t')))
                rid=link.get(qn('r:id'))
                if rid and label:text+=' ['+label+']('+str(doc.part.rels[rid].target_ref)+')'
            if not text or text=='Contents' or text.startswith('Contents are updated'):
                continue
            if style=='Code':
                if not in_code:lines.append('```bash');in_code=True
                lines.append(par.text)
            elif style=='Title':lines.extend(['# '+text,''])
            elif style.startswith('Heading '):lines.extend(['#'*(int(style[-1])+1)+' '+text,''])
            else:lines.extend([text,''])
        elif element.tag==qn('w:tbl'):
            if in_code:lines.extend(['```','']);in_code=False
            tab=Table(element,doc)
            for i,row in enumerate(tab.rows):
                vals=[c.text.replace('\n','<br>').replace('|','\\|') for c in row.cells]
                lines.append('| '+' | '.join(vals)+' |')
                if i==0:lines.append('| '+' | '.join(['---']*len(vals))+' |')
            lines.append('')
    if in_code:lines.extend(['```',''])
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def build():
    before={name:digest(ROOT/'results'/name) for name in INPUTS}
    figures()
    doc=Document();configure_report(doc)
    authors=json.loads((ROOT/'assets/authors.json').read_text())
    # A restrained title page; technical summaries belong in the main text.
    par=p(doc,'');par.paragraph_format.space_before=Pt(55);par.paragraph_format.space_after=Pt(18)
    border=OxmlElement('w:pBdr');line=OxmlElement('w:bottom')
    for k,v in [('val','single'),('sz','8'),('color','003D7C')]:line.set(qn('w:'+k),v)
    border.append(line);par._p.get_or_add_pPr().append(border)
    par=p(doc,'Parallel K-means clustering with MPI and CUDA','Title');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after=Pt(14)
    for run in par.runs:run.font.size=Pt(16)
    par=p(doc,'Distributed centroid reduction and GPU acceleration on Atlas and Vanda','Subtitle');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after=Pt(25)
    par=p(doc,'');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par.add_run().add_picture(str(ROOT/'assets/nus_logo_full-vertical.jpg'),width=Cm(4.3))
    par.paragraph_format.space_after=Pt(32)
    par=p(doc,'DSA5208 · Scalable Distributed Computing for Data Science');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par.runs[0].bold=True
    author_table=table(doc,['AUTHOR','STUDENT ID'],[(a['name'],a['student_id']) for a in sorted(authors,key=lambda a:a['name'])],[10.3,6.1],size=9.9)
    for row in author_table.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par=p(doc,'National University of Singapore · November 2026');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par=p(doc,'Source code and reproducible experiment records');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    par=p(doc,'');par.alignment=WD_ALIGN_PARAGRAPH.CENTER
    hyperlink(par,'github.com/AmritSatpathy/parallelKmean','https://github.com/AmritSatpathy/parallelKmean')
    doc.add_page_break()
    h(doc,'Contents')
    doc.paragraphs[-1].paragraph_format.space_after=Pt(24)
    toc=p(doc,'');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'TOC \\o "1-1" \\h \\z \\u')
    rr=OxmlElement('w:r');tt=OxmlElement('w:t');tt.text='Contents are updated automatically when rendered.';rr.append(tt);field.append(rr);toc._p.append(field)
    doc.add_page_break()
    h(doc,'Abstract')
    p(doc,'K-means combines independent nearest-centroid searches with a globally coupled centroid update, making it a useful case study in distributed and accelerator computing. We implement Lloyd’s algorithm in serial C++, CPU MPI, and MPI+CUDA, sharing the same data representation and k-means++ seeding. MPI partitions points across ranks and combines local centroid statistics through three Allreduce collectives per iteration. CUDA replaces each rank’s local assignment loop with one thread per point while keeping cross-rank reductions on host buffers. Experiments use MNIST for real-data interpretation and synthetic Gaussian blobs for strong scaling. The Atlas 100-iteration workload covers three million 64-dimensional points and 128 clusters at 12, 16, 24, 32, and 48 ranks across six completed jobs. At 48 ranks, clustering takes 36.422506 s versus a job-matched serial baseline of 1722.174826 s, giving 47.28× speedup and 98.51% efficiency. A complementary 32-iteration study reaches 31.32× at 32 ranks across two nodes. Vanda GPU batches A and B both approximately halve runtime from one to two A40s; batch B takes 10.159337 s and 5.073951 s respectively, giving 2.002× speedup. Matching displayed inertia supports numerical agreement within matched workloads. The report integrates completed team experiments and distinguishes iteration-loop timings from initialization and scheduler wall time, while retaining clear limits on numerical identity and run-to-run variability.')
    h(doc,'1. Introduction and objectives')
    p(doc,'The project investigates how decomposition and synchronization affect the parallel scaling of a nontrivial algorithm. Lloyd’s K-means alternates nearest-centroid assignment and centroid recomputation [1]. The assignment step is independent across points, but the update requires statistics from every partition. The complete algorithm therefore requires communication and coordinated iteration, satisfying the assignment’s requirement to go beyond an embarrassingly parallel workload.')
    p(doc,'Our objectives are to establish a serial correctness reference, explain the MPI design and its CUDA extension, measure strong scaling on Atlas and Vanda, and interpret clustering quality on MNIST. The implementation remains within the MPI+CUDA scope. Larger algorithms or additional programming frameworks are unnecessary to answer these questions.')
    p(doc,'The completed experimental program covers CPU scaling through 48 MPI ranks, two fixed-iteration workload budgets, and GPU scaling through two devices in two measurement batches. All runs contribute to the same team study. Workload and batch labels preserve valid baseline comparisons while the execution ledger links the completed experiments to their evidence.')
    h(doc,'2. Algorithm and datasets')
    h(doc,'2.1 Objective, initialization, and iteration',2)
    p(doc,'For n points xᵢ in d dimensions, K centroids μⱼ, and cluster assignment a(i), K-means minimizes the within-cluster sum of squared distances, or inertia. The squared distance avoids a square root without changing the nearest-centroid decision.')
    native_equation(doc,[('J = ∑', 'i=1…n',None),(' ‖x', 'i',None),(' − μ','a(i)',None),('‖',None,'2')],1)
    p(doc,'All variants use k-means++ initialization [2], with a fixed seed for comparisons within a series. Each iteration assigns points to their nearest centroid, accumulates sums and counts, and replaces each nonempty centroid by the arithmetic mean of its assigned points. Empty clusters retain their previous centroid. Assignment costs O(nKd); centroid accumulation costs O(nd). K-means++ seeding costs O(nKd) overall and is performed once outside the timed clustering loop.')
    p(doc,'The default convergence path tests relative inertia change; the CPU implementations also test whether assignments stopped moving. Benchmark runs instead pass --fixed-iter so that all configurations execute the same number of iterations. The log phrase “converged in” is therefore interpreted as “completed” for these fixed-iteration measurements, without claiming that a convergence criterion was met.')
    h(doc,'2.2 Dataset definitions and comparability',2)
    tbl(doc,1,'Data used for quality analysis and performance experiments.',
        ['Dataset / experiment','Points','Features','K','Iterations'],[
        ['MNIST, clustering quality','70,000','784','10','Convergence'],
        ['Synthetic, CPU scaling / GPU batch A','3,000,000','64','128','32 fixed'],
        ['Synthetic, Atlas 12–48 ranks','3,000,000','64','128','100 fixed'],
        ['Synthetic, GPU batch B','3,000,000','64','128','32 fixed'],
        ['MNIST, Vanda smoke test','70,000','784','10','10 fixed']], [6.7,2.4,2.3,1.6,3.4])
    p(doc,'MNIST combines 70,000 digit images, each flattened into 784 raw pixel features; column 0 is excluded from features and retained as a ground-truth digit label [3]. Synthetic Gaussian blobs contain planted cluster labels, again in column 0. The generator supplies controllable n, d, K, seed, and separation, making it suitable for a fixed-workload scaling experiment.')
    note(doc,'Workload comparability.','The 100-iteration Atlas workload reports inertia 1.214576×10⁹, whereas Vanda synthetic runs report approximately 8.624057×10⁸. The Atlas and Vanda datasets downloaded on 12 November are byte-identical by SHA-256; both copies contain three million rows. Historical jobs did not record contemporaneous input hashes. The iteration budgets differ, so these timings are not pooled into a controlled cross-platform speedup.')
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
    tbl(doc,2,'Project execution environments by experiment group.',
        ['Experiment group','Platform / configuration','Toolchain / evidence'],[
        ['Local runs','Ryzen 7 6800HS; 8 cores / 16 threads; RTX 3050 Ti','GCC 13; OpenMPI 4.1; project records'],
        ['Atlas scaling','1-node 16-rank and 2-node 32-rank series','GCC 13.2; OpenMPI 4.1.6; project records'],
        ['Vanda batch A','1–2 NVIDIA A40 GPUs','GCC 13.2; OpenMPI 4.1.6; CUDA 12.4'],
        ['Atlas 100 iterations','12, 16, 24, 32, 48 ranks; one node per job','GCC 7.3 / OpenMPI 4.1.1; retained execution logs'],
        ['Vanda batch B','GN-A40-021 and GN-A40-035; 1–2 A40s','GCC 13.2; OpenMPI 4.1.6; CUDA 12.4 in logs']],
        [3.2,6.5,6.7])
    p(doc,'The CPU harness runs serial before the MPI configurations. REPS controls repetition and retains the minimum successful time, while fixed iterations ensure comparable iteration counts. The extended Atlas study retains one timing pair per job, including two independent 16-rank jobs. Vanda batch B retains one timing per configuration. Historical repetition counts are not inferred when the retained CSV does not encode them. MPI barriers bracket the distributed timed loop; the GPU loop also synchronizes CUDA work before completing an iteration.')
    p(doc,'Reported time is the clustering loop, including per-iteration computation, host/device copies where applicable, and collective communication. PBS wall time additionally includes build, data generation, CSV read, initialization, process launch, and other job overhead. These are different performance measures. For Vanda batch B, CSV loading alone takes 24.840 s on one GPU and 23.348 s on two, outside the reported 10.159337 s and 5.073951 s clustering times.')
    h(doc,'5. Correctness and real-data interpretation')
    h(doc,'5.1 Numerical agreement within each experiment series',2)
    tbl(doc,3,'Consolidated project validation evidence, with its comparison protocol.',
        ['Evidence','Comparison','Observed result'],[
        ['MNIST label validation','Serial versus 8-rank MPI','100% label agreement; inertia 1.784862×10¹¹; 13 iterations'],
        ['Small synthetic blobs','Serial versus 2/4/8-rank MPI','99.7–99.9% label agreement; inertia difference ~10⁻⁶'],
        ['Large synthetic blobs','Serial / MPI / MPI+CUDA','Inertia 8.624057–8.624058×10⁸'],
        ['Atlas, 100 iterations','Serial versus 12/16/24/32/48 ranks','All twelve outputs print inertia 1.214576×10⁹'],
        ['Vanda batch B, 32 iterations','1 versus 2 GPUs','Both print inertia 8.624057×10⁸'],
        ['GPU MNIST smoke test','1 GPU, 10 iterations','Inertia 1.785707×10¹¹; exit status 0 (session record)']],
        [4.1,4.9,7.4])
    p(doc,'The label comparisons and purity tests are recorded in the project study; their label files and raw validation transcripts are not present in this checkout. Additional inertia checks are supported by the retained CSV and log records. Equality at printed precision supports numerical agreement but does not prove bitwise-equal centroids or labels. Floating-point addition is non-associative, so MPI reduction order and CUDA atomic order can shift boundary assignments [4], [5].')
    p(doc,'Figure 3 uses the newly retained per-iteration Vanda log lines. Both GPU trajectories drop rapidly and then approach the same displayed objective. This provides evidence that parallel execution preserves the iterative behavior, in addition to matching the final diagnostic. The inertia is evaluated during assignment before the subsequent centroid update, matching the source implementation.')
    fig(doc,'06_gpu_inertia',3,'Recorded objective trajectories for Vanda jobs 1533324 and 1533344: solid blue with circles (1 GPU), dashed orange with diamonds (2 GPUs). Markers are staggered for visibility; the nearly overlapping values show within-series numerical agreement.')
    h(doc,'5.2 Conclusions about MNIST',2)
    p(doc,'The project reports approximately 0.585 cluster purity for K=10 on raw MNIST pixels. Purity is the sum of each predicted cluster’s largest true-label count divided by n. Reported majority fractions are approximately 0.94 for a 0-dominated cluster, 0.91 for a 2-dominated cluster, and 0.86 for a 6-dominated cluster. Stroke-similar digits such as 4/7/9 and 3/5/8 are confused. These observations are clustering results, not supervised classification accuracy.')
    p(doc,'Raw-pixel Euclidean distance captures broad shape but is sensitive to position and handwriting variation. Ten digit labels do not force an unsupervised method to discover ten perfectly corresponding groups. The synthetic quality study reports 0.977 purity with 126 of 128 clusters perfectly pure. Feature transformations or alternative clustering methods are outside this project’s MPI+CUDA scope.')
    h(doc,'6. Results: CPU and GPU strong scaling')
    p(doc,'The project’s strong-scaling experiments evaluate CPU ranks and GPU devices while fixing the workload within each comparison. All measurements are retained in the project results collection. CPU results are grouped by allocation, and GPU measurements use batch labels to identify their respective one-GPU baselines. The consolidated scheduler ledger in Appendix A connects available job identifiers to their evidence.')
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
    tbl(doc,4,'MPI clustering runtimes and two-node-allocation scaling (32 fixed iterations).',
        ['Ranks','Local (s)','Atlas 1-node (s)','Atlas 2-node (s)','Speedup¹','Efficiency¹'],rows,
        [1.3,3.0,3.3,3.4,2.6,2.8])
    p(doc,f'¹ Two-node-allocation metrics use its own serial baseline of {baseb:.6f} s. The corresponding baselines are {basel:.6f} s locally and {basea:.6f} s for the single-node Atlas allocation. The 1-rank row above is MPI, not serial. The project records describe ranks filling one node through 16, then both nodes at 32; per-rank placement logs were not retained.')
    fig(doc,'03_original_cpu_scaling',4,'CPU strong scaling with matched serial baselines. The workstation shows a stronger efficiency decline at 16 ranks.')
    p(doc,'The two-node Atlas allocation reaches 31.3216× speedup at 32 ranks and 97.88% efficiency. The single-node allocation reaches 15.7335× at 16 ranks. Locally, speedup is 6.8116× at eight ranks and 9.1431× at 16 ranks. The workstation’s eight physical cores and 16 logical threads make shared execution resources a plausible contributor to the local efficiency decline; the timings do not separately measure SMT, bandwidth, or communication costs.')
    h(doc,'6.2 CUDA scaling: measurement batch A',2)
    old=read('scaling_vanda_multi.csv');t1,t2=[float(r['time']) for r in old]
    tbl(doc,5,'GPU measurement batch A and the local accelerator reference.',
        ['Configuration / source','Time (s)','Within-series comparison'],[
        ['Local RTX 3050 Ti (project report)','35.2','~16.98× versus local serial 597.873896 s'],
        ['Vanda 1 A40, standalone CSV','8.627849','Independent earlier 1-GPU measurement'],
        ['Vanda 1 A40, paired-series CSV',f'{t1:.6f}','GPU baseline'],
        ['Vanda 2 A40s, paired-series CSV',f'{t2:.6f}',f'{t1/t2:.4f}× speedup; {100*t1/t2/2:.2f}% efficiency']],
        [7.1,3.1,6.2])
    p(doc,'The project’s cross-platform reference compares the two-GPU time of 4.309527 s with the Atlas serial baseline of 404.566169 s, giving approximately 93.88×. This is a cross-platform comparison: hardware, compiler, and placement differ, so it does not isolate processor count. GPU scaling metrics use the matched one-GPU baseline. The local 35.2 s GPU time is recorded in the project report rather than a retained standalone result log.')
    h(doc,'7. Results: extended workload and GPU verification')
    h(doc,'7.1 Atlas strong scaling: 100 fixed iterations',2)
    runs=atlas_metrics()
    tbl(doc,6,'Completed Atlas 100-iteration jobs; each speedup uses the serial baseline from the same job.',
        ['PBS job ID','Ranks','Serial (s)','MPI (s)','Speedup','Efficiency'],
        [[r['job_id'],r['ranks'],f"{r['serial_time']:.6f}",f"{r['mpi_time']:.6f}",
          f"{r['speedup']:.4f}×",f"{100*r['efficiency']:.2f}%"] for r in runs],
        [3.6,1.4,3.1,3.1,2.5,2.7])
    fig(doc,'05_atlas_full_load',5,'One project scaling series across six completed jobs at 12–48 ranks. Dots retain individual MPI measurements; connecting lines and runtime labels use the median at repeated rank counts. Each speedup uses its own job’s serial baseline; runtime is logarithmic.')
    latest=[r for r in runs if r['job_id'] in ['1036952.venus01','1037018.venus01','1037036.venus01']]
    p(doc,'The added 12-, 16-, and 24-rank jobs take '+', '.join(f"{r['mpi_time']:.6f} s" for r in latest)+' respectively. Their job-matched speedups are '+', '.join(f"{r['speedup']:.4f}×" for r in latest)+', with efficiencies of '+', '.join(f"{100*r['efficiency']:.2f}%" for r in latest)+'. The earlier completed 16-rank result remains in Table 6 alongside the new measurement. Every serial and MPI row reports 3,000,000 points, 64 dimensions, K=128, 100 iterations, and inertia 1.214576×10⁹. All six PBS logs confirm exit status 0 and one allocated node per job.')
    p(doc,'The new jobs ran on cnode-33-43-26 (12 ranks) and cnode-33-43-53 (16 and 24 ranks), with PBS wall times 00:35:22, 00:32:02, and 00:31:30 respectively. Earlier 16-, 32-, and 48-rank jobs have wall times 00:32:56, 00:36:17, and 00:31:13. These include serial baselines and setup; they are not MPI clustering times. Job-matched serial baselines span 1694.752549–1728.390679 s. Efficiencies slightly above 100% are retained as measured and may reflect cache or timing variation; these few observations do not establish superlinear scaling.')
    h(doc,'7.2 CUDA scaling: measurement batch B',2)
    vr=read('scaling_vanda_multi_latest.csv');v1,v2=[float(r['time']) for r in vr];vsp=v1/v2
    tbl(doc,7,'Vanda GPU measurement batch B, 8 November 2026; 32 fixed iterations.',
        ['PBS job ID','Node','GPUs / ranks','Time (s)','GPU speedup'],[
        ['1533324.stdct-mgmt-02','GN-A40-021','1 / 1',f'{v1:.6f}','1.0000'],
        ['1533344.stdct-mgmt-02','GN-A40-035','2 / 2',f'{v2:.6f}',f'{vsp:.4f}']],
        [5.4,3.4,2.6,2.8,2.2])
    fig(doc,'04_gpu_series',6,'Project GPU scaling across two measurement batches. Each speedup uses the corresponding one-GPU baseline; all recorded timings remain visible.')
    p(doc,f'Batch B achieves {vsp:.4f}× GPU speedup, with {100*vsp/2:.2f}% efficiency. This is approximately ideal scaling; one observation per configuration cannot establish statistically significant superlinear performance. Both runs print final inertia 8.624057×10⁸ and the terminal session records exit status 0. The supplied log copies contain complete iteration trajectories but omit PBS footer summaries; scheduler wall times of 00:01:43 and 00:00:53 are therefore attributed to the recorded qstat output, not inferred from the shorter log files.')
    p(doc,'The logs confirm GCC 13.2.0, OpenMPI 4.1.6, CUDA 12.4, and compilation for sm_86. They also contain nonfatal OpenMPI warnings that libpbs.so.0 could not be loaded for the TM launcher and resource components; those messages were explicitly ignored and the one-node runs produced results. Successful local execution does not establish functioning multi-node PBS integration. Host memory figures reported by PBS are not GPU VRAM utilization.')
    h(doc,'8. Discussion, limitations, and conclusion')
    h(doc,'8.1 What the measurements establish',2)
    p(doc,'The Atlas scaling curves and extended workload show that local assignment work can dominate collective overhead at the tested problem size. The small centroid statistics explain why communication can be amortized over millions of points. Both GPU measurement batches support the same decomposition across two accelerators. These measurements support the design, while a communication-only profile would be needed to attribute exact percentages of runtime to Allreduce, computation, or memory movement.')
    p(doc,'K-means is not embarrassingly parallel as a whole: every centroid update depends on globally reduced statistics and precedes the next assignment. The CUDA implementation keeps that dependency intact while increasing local throughput. The six completed 100-iteration jobs show near-ideal measured scaling over 12–48 ranks, with efficiencies of approximately 98.4–101.6%. These results establish performance over the measured range; additional evidence would be needed to extend the conclusion to larger configurations.')
    h(doc,'8.2 Limits of evidence and interpretation',2)
    p(doc,'Only strong scaling is measured. Completed CPU experiments cover up to 48 ranks and GPU experiments up to two A40s on one node. The retrieved Atlas and Vanda inputs have matching checksums. Historical job-time hashes and complete build manifests were not retained. Similar printed inertia is a useful diagnostic rather than a complete proof of identical cluster assignments. Local runs are identified by their execution platform and recorded results rather than PBS job IDs.')
    p(doc,'The extended Atlas study contains two independent 16-rank jobs and one job at each other rank count; GPU batch B has one timing per configuration. These observations do not provide confidence intervals. The harness selects a minimum when repeated, but raw repetitions are not stored in the CSV. Comparisons are measured examples under shared-cluster conditions. Scheduler allocation counts may exceed active MPI ranks: Vanda assigned 36 CPU slots for one GPU and 72 for two in the session record, while the code launched only one or two ranks.')
    h(doc,'8.3 Conclusion',2)
    p(doc,'The project delivers serial, MPI CPU, and MPI+CUDA implementations of the same Lloyd iteration, with explicit global centroid reductions and completed HPC timing evidence. The Atlas 100-iteration workload covers 12, 16, 24, 32, and 48 ranks across six completed jobs. It reaches 47.28× speedup at 48 ranks, with measured efficiencies spanning approximately 98.4–101.6% across the study. The complementary 32-iteration study reaches 31.32× at 32 ranks across two nodes. Both Vanda measurement batches achieve approximately 2.00× speedup from one to two A40s. MNIST illustrates the limits of raw-pixel clustering, while numerical comparisons support correctness within matched workloads. The principal contribution is the measured relationship between point-level parallel work and the synchronized centroid update, supported by the team’s completed execution records.')
    h(doc,'References')
    refs=[
        ('S. Lloyd, “Least squares quantization in PCM,” IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 129–137, 1982.','https://doi.org/10.1109/TIT.1982.1056489'),
        ('D. Arthur and S. Vassilvitskii, “k-means++: The advantages of careful seeding,” Proc. 18th ACM–SIAM Symposium on Discrete Algorithms, pp. 1027–1035, 2007.','https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf'),
        ('Y. LeCun, C. Cortes, and C. J. C. Burges, “The MNIST database of handwritten digits.” Dataset source.','https://yann.lecun.com/exdb/mnist/'),
        ('Message Passing Interface Forum, MPI: A Message-Passing Interface Standard, version 4.1, November 2023. Standard reference; deployed OpenMPI versions are separately recorded in Table 2.','https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf'),
        ('NVIDIA, CUDA C++ Programming Guide, toolkit version 12.4, 2024.','https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html'),
        ('National University of Singapore, University Policy Workgroup for AI in Teaching and Learning, Policy for the Use of AI in Teaching and Learning, current version 15 September 2026, §§2.5–2.6, 3–4. Accessed 8 November 2026.','https://ctlt.nus.edu.sg/wp-content/uploads/2026/04/Policy-for-Use-of-AI-in-Teaching-and-Learning.pdf'),
        ('NUS Libraries, “Citation Styles: Citing AI Tools,” updated 28 August 2026. Accessed 8 November 2026.','https://libguides.nus.edu.sg/citation/ai'),
        ('AI acknowledgment. The team used GPT 6.1 Astra, GPT 6.1 Solo, GPT 6 Astra, Claude, ChatGPT/Codex, and Kiro for coding, HPC troubleshooting, analysis, diagrams, and writing. Both authors reviewed the work and retain full responsibility for its accuracy and submission, following NUS guidance [6], [7].',None),
        ('A. Satpathy and R. Srivastava, Parallel K-means project repository: source, experiment CSV files, consolidated HPC logs, and report, 2026. Referenced throughout Appendices A–B.','https://github.com/AmritSatpathy/parallelKmean')]
    for i,(text,url) in enumerate(refs,1):
        par=p(doc,f'[{i}] {text} ','Reference')
        if url:hyperlink(par,'Online source',url)
    h(doc,'Appendix A. Experiment provenance and scheduler ledger')
    p(doc,'This ledger consolidates the team’s completed local and HPC experiments used in the report [9]. Local runs are identified by platform and configuration. Completed HPC jobs retain their documented scheduler identifiers. Artifact checksums in report/experiment_manifest.json identify the retained CSV and log evidence.')
    tbl(doc,'A1','Completed project measurements by execution platform.',
        ['Measurement artifact / record','Coverage','Execution'],[
        ['scaling_big.csv','Serial + MPI ranks 1–16','Local run — CPU'],
        ['RTX 3050 Ti result: 35.2 s','One-GPU CUDA reference','Local run — CUDA'],
        ['scaling_atlas.csv','Serial + MPI ranks 1–16','Atlas CPU'],
        ['scaling_atlas_2node.csv','Serial + MPI ranks 1–32','Atlas CPU, two-node allocation'],
        ['scaling_vanda_gpu.csv','Standalone one-GPU result','Vanda GPU'],
        ['scaling_vanda_multi.csv','Batch A: one and two GPUs','Vanda GPU']],
        [6.4,4.7,5.3])
    tbl(doc,'A2','Project scheduler ledger: successful runs with retained job identifiers.',
        ['Job identifier','Purpose / outcome','Evidence retained'],[
        ['1027948.venus01','Atlas MNIST sweep; exit 0; 00:04:33','pbs/logs/atlas/1027948.venus01.OU'],
        ['1034257.venus01','Atlas synthetic serial + 16 ranks; 100 iterations; exit 0','pbs/logs/1034257.venus01.OU'],
        ['1036751.venus01','Atlas serial + 32 ranks; 100 iterations; exit 0','pbs/logs/atlas/1036751.venus01.OU'],
        ['1036776.venus01','Atlas serial + 48 ranks; 100 iterations; exit 0','pbs/logs/atlas/1036776.venus01.OU'],
        ['1036952.venus01','Atlas serial + 12 ranks; 100 iterations; exit 0','pbs/logs/atlas/1036952.venus01.OU'],
        ['1037018.venus01','Atlas serial + 16 ranks; 100 iterations; exit 0','pbs/logs/atlas/1037018.venus01.OU'],
        ['1037036.venus01','Atlas serial + 24 ranks; 100 iterations; exit 0','pbs/logs/atlas/1037036.venus01.OU'],
        ['1533323.stdct-mgmt-02','Vanda MNIST test; 10 iterations; exit 0','Completed session record'],
        ['1533324.stdct-mgmt-02','Vanda 1 A40; 32 iterations; exit 0','pbs/logs/vanda/vanda_1gpu_full.OU'],
        ['1533344.stdct-mgmt-02','Vanda 2 A40s; 32 iterations; exit 0','pbs/logs/vanda/vanda_2gpu_full.OU']],
        [5.1,6.0,5.3])
    p(doc,'The Vanda log copies were renamed descriptively during download. Their full PBS identifiers are preserved from the terminal’s original log filenames: 1533324.stdct-mgmt-02.OU and 1533344.stdct-mgmt-02.OU. The log bodies corroborate node, ranks, toolkit, iteration count, inertia, and runtime; they do not themselves contain a JobId field.')
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
    code(doc,'mkdir -p bin data results pbs/logs\nqsub -l select=1:ncpus=48:mpiprocs=48:mem=40gb \\\n+  -l walltime=02:00:00 \\\n+  -v "CSV=data/blobs_big.csv,K=128,PROCS=48,MAXIT=100,REPS=1" \\\n+  pbs/atlas_sweep.pbs')
    p(doc,'For other completed CPU configurations, replace all three occurrences of 48 above with 12, 16, 24, or 32. Use a separate working directory per job to preserve binaries and result files. On Atlas 9 the working script uses explicit GCC 7.3 and OpenMPI 4.1.1 paths and a clean library path. Compilation uses the compute node’s CPU flags, including -march=native; binaries are rebuilt inside the scheduled job.')
    code(doc,'module load GCC/13.2.0 OpenMPI/4.1.6-GCC-13.2.0 CUDA/12.4.0\nqsub -l select=1:ncpus=72:ngpus=2:mpiprocs=2:mem=480gb \\\n+  -l walltime=01:00:00 \\\n+  -v "NP=2,CSV=data/blobs_big.csv,K=128,MAXIT=32,SEED=42,CUDA_ARCH=sm_86" \\\n+  pbs/vanda_cuda.pbs')
    p(doc,'For one GPU use ncpus=36, mem=240gb, ngpus=1, mpiprocs=1, and NP=1, matching the recorded Vanda allocation. Both GPU counts should use the same dataset file and seed. PBS directives must not have trailing inline comments. Submit from the repository root and pre-create pbs/logs and data; compilation and large data generation run inside the allocation.')
    h(doc,'B.3 Validation and job inspection',2)
    code(doc,'python3 scripts/compare_labels.py results/serial.labels results/mpi.labels\npython3 scripts/validate.py --pred results/mpi.labels \\\n+  --csv data/mnist.csv --truth-col 0\nqstat -x -f JOB_ID\ngrep "^RESULT" pbs/logs/JOB_ID*.OU')
    p(doc,'A completed benchmark is checked using exit status 0 together with its RESULT or CSV rows, expected n/d/K, rank count, iteration count, and final inertia. Each of the six 100-iteration Atlas jobs contains its serial and MPI rows, and the Vanda logs contain the expected one- and two-GPU results.')
    h(doc,'B.4 Data preservation and report build',2)
    p(doc,'The report builder reads CSV files and logs but does not modify them or submit jobs. Generated charts are stored under report/figures_publication/ as PNG and SVG. report/experiment_manifest.json records artifact hashes and known run IDs. The editable report is report/Parallel_Kmeans_Project_Report.docx, and its rendered counterpart is report/Parallel_Kmeans_Project_Report.pdf. The Markdown source report/report.md also records the project’s extended experiments.')
    p(doc,'To keep the submission small, dataset CSVs are excluded. The README provides MNIST download and checksum-verification commands, plus synthetic-data generation instructions. Public MNIST files match the project inputs; retrieved Atlas and Vanda synthetic copies are byte-identical. Data checksums and HPC source snapshots are retained. Record dataset hashes and distinct experiment filenames in future runs. Workload and measurement-batch labels support valid comparisons across the completed project archive: the 100-iteration Atlas values must not replace a 32-iteration baseline. Local runs retain their platform and configuration labels, while completed HPC records retain their job identifiers.')
    # Explicit table geometry is re-applied after every row exists.
    for t in doc.tables:
        widths=[int(c.get(qn('w:w'))) for c in t._tbl.tblGrid]
        for row in t.rows:
            for cell,w in zip(row.cells,widths):
                cell._tc.get_or_add_tcPr().find(qn('w:tcW')).set(qn('w:w'),str(w))
        pr=t._tbl.tblPr;indent=OxmlElement('w:tblInd');indent.set(qn('w:w'),'80');indent.set(qn('w:type'),'dxa');pr.append(indent)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            node=OxmlElement('w:'+edge)
            for key,value in [('val','single'),('sz','4'),('color','AAB3BC')]:
                node.set(qn('w:'+key),value)
            borders.append(node)
        pr.append(borders)
    dest=OUT/'Parallel_Kmeans_Project_Report.docx';doc.save(dest)
    write_markdown(doc)
    after={name:digest(ROOT/'results'/name) for name in INPUTS}
    assert before==after,'Experimental inputs changed unexpectedly'
    records=[]
    for file in INPUTS:
        groups={'scaling_big.csv':'local_cpu_scaling','scaling_atlas.csv':'atlas_single_node_scaling',
                'scaling_atlas_2node.csv':'atlas_two_node_scaling','scaling_vanda_gpu.csv':'vanda_single_gpu_reference',
                'scaling_vanda_multi.csv':'vanda_gpu_batch_a','scaling_full_16cores_100iter.csv':'atlas_100_iteration_workload',
                'scaling_vanda_multi_latest.csv':'vanda_gpu_batch_b',
                'scaling_32cores_100iter_1036751.csv':'atlas_100_iteration_32_ranks',
                'scaling_48cores_100iter_1036776.csv':'atlas_100_iteration_48_ranks',
                'scaling_12cores_100iter_1036952.csv':'atlas_100_iteration_12_ranks',
                'scaling_16cores_100iter_1037018.csv':'atlas_100_iteration_16_ranks',
                'scaling_24cores_100iter_1037036.csv':'atlas_100_iteration_24_ranks'}
        records.append({'path':'results/'+file,'sha256':before[file],
                        'experiment_group':groups[file], 'rows':read(file)})
    logs=[r['log'] for r in ATLAS_RUNS]+['pbs/logs/atlas/1027948.venus01.OU','pbs/logs/vanda/vanda_1gpu_full.OU','pbs/logs/vanda/vanda_2gpu_full.OU']
    manifest={'title':'Parallel K-means report evidence','report_date':'2026-11-12',
              'authors':authors,'original_inputs_preserved':True,'completed_runs_only':True,'experiments':records,
              'logs':[{'path':f,'sha256':digest(ROOT/f)} for f in logs],
              'documented_run_ids':['1027948.venus01',*[r['job_id'] for r in ATLAS_RUNS],
                                    '1533323.stdct-mgmt-02','1533324.stdct-mgmt-02','1533344.stdct-mgmt-02'],
              'atlas_100_iteration_metrics':atlas_metrics(),
              'local_runs':[{'execution':'Local run — CPU','source':'results/scaling_big.csv'},
                            {'execution':'Local run — CUDA','device':'RTX 3050 Ti','time_s':35.2,'iterations':32}],
              'interpretation':'One team project; retrieved Atlas and Vanda synthetic datasets are byte-identical by SHA-256; historical job-time hashes were not recorded; CPU/GPU comparisons require matched iteration budgets.'}
    (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest)
    print('Original result files unchanged; 6 figures generated; provenance manifest written.')


if __name__=='__main__':
    build()
