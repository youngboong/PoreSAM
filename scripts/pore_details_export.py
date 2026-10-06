"""Standalone scientific plot export from the same selected pore measurements."""
import base64
import io
import math
import os
from pathlib import Path
import numpy as np
from analyze_candidates import measure_masks
from pore_extra_metrics import LABELS as EXTRA_LABELS

LABELS={'length_um':'Length (µm)','width_um':'Width (µm)','aspect_ratio':'Aspect ratio',
        'equivalent_diameter_um':'Equivalent diameter (µm)','roundness':'Roundness',
        'area_um2':'Area (µm²)','circularity':'Circularity','image_area_percent':'Area fraction (%)',**EXTRA_LABELS}

KOREAN_LABELS={
    'length_um':'길이 (µm)', 'width_um':'너비 (µm)', 'aspect_ratio':'종횡비',
    'equivalent_diameter_um':'등가원직경 (µm)', 'roundness':'원형도',
    'area_um2':'면적 (µm²)', 'circularity':'원형도 (둘레 기준)', 'image_area_percent':'면적분율 (%)',
    'angle_deg':'각도 (°)', 'area_box_ratio':'면적 / 경계 상자 면적',
    'brightness_max':'최대 밝기', 'brightness_mean':'평균 밝기',
    'brightness_min':'최소 밝기', 'brightness_std':'밝기 표준편차',
    'centroid_x_um':'중심 X (µm)', 'centroid_y_um':'중심 Y (µm)',
    'convexity':'볼록도 (둘레 기준)', 'integral_density':'밝기 합계',
    'mass_center_x_um':'밝기 가중 중심 X (µm)', 'mass_center_y_um':'밝기 가중 중심 Y (µm)',
    'perimeter_um':'둘레 (µm)', 'rectangle_bottom_um':'경계 상자 아래쪽 (µm)',
    'rectangle_left_um':'경계 상자 왼쪽 (µm)', 'rectangle_right_um':'경계 상자 오른쪽 (µm)',
    'rectangle_top_um':'경계 상자 위쪽 (µm)', 'solidity':'볼록 영역 면적비',
}


def integer_histogram_edges(values, target_bins):
    lo=math.floor(float(np.min(values)))
    hi=float(np.max(values))
    width=max(1,math.ceil((hi-lo)/target_bins))
    count=max(1,math.ceil((hi-lo)/width))
    return lo+np.arange(count+1)*width


def report_histogram_edges(values, target_bins):
    lo=math.floor(float(np.min(values))*2)/2
    hi=float(np.max(values))
    width=max(.5,math.ceil((hi-lo)/target_bins*2)/2)
    count=max(1,math.ceil((hi-lo)/width))
    return lo+np.arange(count+1)*width


UNIT_INTERVAL_KEYS = {'roundness','circularity','solidity','convexity','area_box_ratio'}

def draw_report_histogram(ax, values, key, label, bins='auto'):
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    ax.tick_params(direction='in',top=True,right=True,width=1.2,labelsize=12)
    for spine in ax.spines.values():spine.set_linewidth(1.4)
    ax.grid(False)
    values=np.asarray(values,dtype=float)
    if bins=='auto':bins=min(30,max(10,math.ceil(2*math.sqrt(len(values)))))
    elif type(bins) is not int or bins not in [5,10,20,30]:raise ValueError('Select a valid bin count.')
    if len(values):
        if key in UNIT_INTERVAL_KEYS:
            # Dimensionless 0–1 measurements need fractional bins to retain shape information.
            count=min(20,bins)
            edges=np.linspace(0,1,count+1)
        else:edges=report_histogram_edges(values,bins)
        counts,_,_=ax.hist(values,bins=edges,facecolor='white',edgecolor='#e12626',hatch=chr(92)*3,linewidth=.8,rwidth=1.0)
        stride=max(1,math.ceil((len(edges)-1)/6))
        ax.set_xticks(sorted(set(edges[::stride].tolist()+[float(edges[-1])])))
        ax.set_xlim(edges[0],edges[-1])
        ax.set_ylim(0,max(float(counts.max())*1.12,1))
    else:
        ax.text(.5,.5,'경계에 닿지 않는 pore가 없습니다.',transform=ax.transAxes,ha='center')
        ax.set_ylim(0,1)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True,prune='lower'))
    ax.set_xlabel(label,fontsize=14,fontweight='bold')
    ax.set_ylabel('개수',fontsize=14,fontweight='bold')
    ax.text(.98,.96,f'n = {len(values)}',transform=ax.transAxes,ha='right',va='top',fontsize=12)


def export_plot(state,payload,*,korean=False):
    labels=KOREAN_LABELS if korean else LABELS
    ids=payload.get('candidate_ids')
    if not isinstance(ids,list) or any(type(i) is not int or i not in state['masks'] for i in ids):
        raise ValueError('Select valid pores.')
    kind,x_key,y_key=payload.get('kind'),payload.get('x_key'),payload.get('y_key')
    if kind not in ['histogram','scatter'] or x_key not in LABELS or (kind=='scatter' and y_key not in LABELS):
        raise ValueError('Select valid plot axes.')
    rows,_,*_=measure_masks([(i,state['masks'][i]) for i in sorted(set(ids))],state['gray'].shape,state['report']['scale']['um_per_pixel'],gray=state['gray'])
    rows['image_area_percent']=rows['area_pixels']/state['gray'].size*100
    keys=[x_key,y_key] if kind=='scatter' else [x_key]
    rows=rows.dropna(subset=keys)
    if rows.empty:raise ValueError('No valid values to plot.')
    from app_paths import output_root
    cache=output_root()/'.matplotlib';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if korean:plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(6.4,4.4) if korean and kind=='histogram' else (9,5.5),constrained_layout=True)
    try:
        x=rows[x_key].to_numpy(dtype=float)
        if korean and kind=='histogram':
            draw_report_histogram(ax,x,x_key,labels[x_key],payload.get('bins','auto'))
        elif kind=='histogram':
            bins=payload.get('bins','auto')
            if bins=='auto':bins=min(40,max(1,math.ceil(math.sqrt(len(rows)))))
            elif type(bins) is not int or bins not in [5,10,20,30]:raise ValueError('Select a valid bin count.')
            edges=integer_histogram_edges(x,bins)
            counts,_,_=ax.hist(x,bins=edges,color='#187b74',edgecolor='white')
            stride=max(1,math.ceil((len(edges)-1)/6))
            ax.set_xticks(sorted(set(edges[::stride].tolist()+[int(edges[-1])])))
            ax.set_xlim(edges[0],edges[-1])
            from matplotlib.ticker import MaxNLocator,StrMethodFormatter
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(0,max(counts.max()*1.1,1))
            ax.set(ylabel='개수' if korean else 'Count',title=f'히스토그램 (n={len(rows)})' if korean else f'Histogram (n={len(rows)})')
        else:
            ax.scatter(x,rows[y_key],color='#187b74',edgecolors='white',s=35)
            ax.set(ylabel=labels[y_key],title=f'산점도 (n={len(rows)})' if korean else f'Scatter plot (n={len(rows)})')
        ax.set_xlabel(labels[x_key]);ax.set_axisbelow(True)
        if not (korean and kind=='histogram'):ax.grid(alpha=.2)
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=200)
        return dict(image='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii'),count=len(rows))
    finally:plt.close(fig)
