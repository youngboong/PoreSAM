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


def integer_histogram_edges(values, target_bins):
    lo=math.floor(float(np.min(values)))
    hi=float(np.max(values))
    width=max(1,math.ceil((hi-lo)/target_bins))
    count=max(1,math.ceil((hi-lo)/width))
    return lo+np.arange(count+1)*width


def export_plot(state,payload):
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
    fig,ax=plt.subplots(figsize=(9,5.5),constrained_layout=True)
    try:
        x=rows[x_key].to_numpy(dtype=float)
        if kind=='histogram':
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
            ax.set(ylabel='Count',title=f'Histogram (n={len(rows)})')
        else:
            ax.scatter(x,rows[y_key],color='#187b74',edgecolors='white',s=35)
            ax.set(ylabel=LABELS[y_key],title=f'Scatter plot (n={len(rows)})')
        ax.set_xlabel(LABELS[x_key]);ax.grid(alpha=.2);ax.set_axisbelow(True)
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=200)
        return dict(image='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii'),count=len(rows))
    finally:plt.close(fig)
