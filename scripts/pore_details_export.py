"""Standalone scientific plot export from the same selected pore measurements."""
import base64
import io
import math
import os
from pathlib import Path
import numpy as np
from analyze_candidates import measure_masks

LABELS={'length_um':'Length (µm)','width_um':'Width (µm)','aspect_ratio':'Aspect ratio',
        'equivalent_diameter_um':'Equivalent diameter (µm)','roundness':'Roundness',
        'area_um2':'Area (µm²)','circularity':'Circularity','image_area_percent':'Image area (%)'}


def export_plot(state,payload):
    ids=payload.get('candidate_ids')
    if not isinstance(ids,list) or any(type(i) is not int or i not in state['masks'] for i in ids):
        raise ValueError('Select valid pores.')
    kind,x_key,y_key=payload.get('kind'),payload.get('x_key'),payload.get('y_key')
    if kind not in ['histogram','scatter'] or x_key not in LABELS or (kind=='scatter' and y_key not in LABELS):
        raise ValueError('Select valid plot axes.')
    rows,_,*_=measure_masks([(i,state['masks'][i]) for i in sorted(set(ids))],state['gray'].shape,state['report']['scale']['um_per_pixel'])
    rows['image_area_percent']=rows['area_pixels']/state['gray'].size*100
    keys=[x_key,y_key] if kind=='scatter' else [x_key]
    rows=rows.dropna(subset=keys)
    if rows.empty:raise ValueError('No valid values to plot.')
    cache=Path(__file__).resolve().parents[1]/'outputs/.matplotlib';cache.mkdir(parents=True,exist_ok=True)
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
            lo,hi=float(x.min()),float(x.max())
            if lo==hi:d=max(abs(lo)*.05,.5);lo-=d;hi+=d
            ax.hist(x,bins=np.linspace(lo,hi,bins+1),color='#187b74',edgecolor='white')
            ax.set(ylabel='Count',title=f'Histogram (n={len(rows)})')
        else:
            ax.scatter(x,rows[y_key],color='#187b74',edgecolors='white',s=35)
            ax.set(ylabel=LABELS[y_key],title=f'Scatter plot (n={len(rows)})')
        ax.set_xlabel(LABELS[x_key]);ax.grid(alpha=.2);ax.set_axisbelow(True)
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=200)
        return dict(image='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii'),count=len(rows))
    finally:plt.close(fig)
