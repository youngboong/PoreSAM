(() => {
  const dialog=$('poreDetailsDialog'),plot=$('detailsPlot'),ctx=plot.getContext('2d');
  const metrics=[
    ['length_um','Length (µm)'],['width_um','Width (µm)'],['aspect_ratio','Aspect ratio'],
    ['equivalent_diameter_um','Equivalent diameter (µm)'],['roundness','Roundness'],
    ['area_um2','Area (µm²)'],['circularity','Circularity'],['image_area_percent','Image area (%)']
  ];
  const tableMetrics=metrics.slice(0,5),labels=Object.fromEntries(metrics);
  const format=value=>value==null||!Number.isFinite(value)?'—':value.toLocaleString(undefined,{maximumFractionDigits:3,minimumFractionDigits:3});
  let data=null,selectedIds=new Set(),allSelected=true,view='table',points=[],bars=[],plotRows=[],lastPlot=null;
  const selectedRows=()=>data?data.candidates.filter(row=>selectedIds.has(row.candidate_id)):[];
  const valid=value=>typeof value==='number'&&Number.isFinite(value);
  function statistics(rows,key){
    const values=rows.map(row=>row[key]).filter(valid),n=values.length;
    if(!n)return {n:0,min:null,max:null,mean:null,std:null};
    let min=Infinity,max=-Infinity,mean=0,m2=0;
    values.forEach((v,i)=>{min=Math.min(min,v);max=Math.max(max,v);const delta=v-mean;mean+=delta/(i+1);m2+=delta*(v-mean)});
    return {n,min,max,mean,std:n>1?Math.sqrt(Math.max(0,m2)/(n-1)):null};
  }
  function statRows(rows,columns){
    const fragment=document.createDocumentFragment(),stats=columns.map(([key])=>statistics(rows,key));
    for(const name of ['min','max','mean','std']){
      const row=document.createElement('tr'),heading=document.createElement('th');heading.textContent=name;row.append(heading);
      stats.forEach(stat=>{const cell=document.createElement('td');cell.textContent=format(stat[name]);cell.title='Valid values: '+stat.n+'';row.append(cell)});fragment.append(row);
    }
    return fragment;
  }
  function renderTable(){
    const fragment=document.createDocumentFragment(),highlight=new Set(window.getSelectedPores?.()??[]);
    for(const candidate of selectedRows()){
      const row=document.createElement('tr');row.dataset.id=candidate.candidate_id;row.tabIndex=0;
      row.classList.toggle('selected',highlight.has(candidate.candidate_id));row.setAttribute('aria-selected',String(highlight.has(candidate.candidate_id)));
      const id=document.createElement('td');id.textContent=candidate.candidate_id+(candidate.touches_image_edge?' †':'');
      id.title=({manual_polygon:'Manual polygon',prompted_sam:'Prompted SAM',manual_cut:'Cut',automatic:'Automatic'})[candidate.source]??'Manual edit';row.append(id);
      tableMetrics.forEach(([key])=>{const cell=document.createElement('td');cell.textContent=format(candidate[key]);row.append(cell)});
      row.onclick=event=>window.selectPoreFromDetails?.(candidate.candidate_id,event.ctrlKey);
      row.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();window.selectPoreFromDetails?.(candidate.candidate_id,event.ctrlKey)}};
      fragment.append(row);
    }
    if(!fragment.childNodes.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=6;cell.textContent='No pores selected.';row.append(cell);fragment.append(row)}
    $('poreListRows').replaceChildren(fragment);$('poreDetailsStats').replaceChildren(statRows(selectedRows(),tableMetrics));
  }
  function renderTargets(){
    const fragment=document.createDocumentFragment(),search=$('detailsTargetSearch').value.trim();
    for(const candidate of data?.candidates??[]){
      if(search&&!String(candidate.candidate_id).includes(search))continue;
      const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.value=candidate.candidate_id;check.checked=selectedIds.has(candidate.candidate_id);
      check.onchange=()=>{if(check.checked)selectedIds.add(candidate.candidate_id);else selectedIds.delete(candidate.candidate_id);allSelected=selectedIds.size===data.candidates.length;update()};
      label.append(check,document.createTextNode('ID '+candidate.candidate_id+(candidate.source==='manual_polygon'?' · Manual polygon':'')));fragment.append(label);
    }
    $('detailsTargetOptions').replaceChildren(fragment);
  }
  function update(){
    const rows=selectedRows();$('detailsTargetCount').textContent='('+rows.length+'/'+(data?.candidates.length??0)+')';
    $('detailsSummary').textContent='Selected: '+rows.length+' · Image area fraction (2D): '+format(data?.stats.candidate_union_area_percent)+'%';
    const highlighted=window.getHighlightedPore?.();if(highlighted!=null&&!selectedIds.has(highlighted))window.clearPoreHighlight?.();
    renderTable();if(view!=='table')drawPlot();
  }
  function extent(values){
    let lo=Math.min(...values),hi=Math.max(...values);
    if(lo===hi){const d=Math.max(Math.abs(lo)*.05,.5);lo-=d;hi+=d;}
    return [lo,hi];
  }
  const frame={left:100,top:68,width:950,height:455};
  function axes(xRange,yRange,xLabel,yLabel,title){
    ctx.fillStyle='#ffffff';ctx.fillRect(0,0,plot.width,plot.height);
    ctx.fillStyle='#253c43';ctx.font='bold 22px sans-serif';ctx.textAlign='center';ctx.fillText(title,plot.width/2,32);
    ctx.font='16px sans-serif';
    for(let i=0;i<=5;i++){
      const x=frame.left+i*frame.width/5,y=frame.top+frame.height-i*frame.height/5;
      ctx.strokeStyle='#e6eded';ctx.beginPath();ctx.moveTo(x,frame.top);ctx.lineTo(x,frame.top+frame.height);ctx.moveTo(frame.left,y);ctx.lineTo(frame.left+frame.width,y);ctx.stroke();
      ctx.fillStyle='#536e74';ctx.textAlign='center';ctx.fillText(Number((xRange[0]+i*(xRange[1]-xRange[0])/5).toPrecision(4)).toString(),x,frame.top+frame.height+27);
      ctx.textAlign='right';ctx.fillText(Number((yRange[0]+i*(yRange[1]-yRange[0])/5).toPrecision(4)).toString(),frame.left-12,y+5);
    }
    ctx.strokeStyle='#647b81';ctx.strokeRect(frame.left,frame.top,frame.width,frame.height);
    ctx.fillStyle='#253c43';ctx.textAlign='center';ctx.fillText(xLabel,frame.left+frame.width/2,plot.height-26);
    ctx.save();ctx.translate(25,frame.top+frame.height/2);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();
  }
  function drawPlot(){
    points=[];bars=[];const xKey=$('detailsXAxis').value,yKey=$('detailsYAxis').value;
    const rows=selectedRows();plotRows=rows.filter(row=>valid(row[xKey])&&(view!=='scatter'||valid(row[yKey])));
    const columns=view==='scatter'?[[xKey,labels[xKey]],[yKey,labels[yKey]]]:[[xKey,labels[xKey]]];
    const header=document.createElement('tr');header.append(document.createElement('th'));
    columns.forEach(([,label])=>{const th=document.createElement('th');th.textContent=label;header.append(th)});
    $('detailsPlotStats').replaceChildren(header,statRows(plotRows,columns));
    $('detailsPlotInfo').textContent='Showing '+plotRows.length+' / Selected: '+rows.length+''+(view==='scatter'?'':'');
    if(!plotRows.length){ctx.fillStyle='white';ctx.fillRect(0,0,plot.width,plot.height);ctx.fillStyle='#64797f';ctx.font='22px sans-serif';ctx.textAlign='center';ctx.fillText('No valid values.',plot.width/2,plot.height/2);lastPlot={type:view,count:0};return;}
    const xRange=extent(plotRows.map(row=>row[xKey]));
    const x=v=>frame.left+(v-xRange[0])/(xRange[1]-xRange[0])*frame.width;
    if(view==='histogram'){
      const n=$('detailsBins').value==='auto'?Math.min(40,Math.max(1,Math.ceil(Math.sqrt(plotRows.length)))):Number($('detailsBins').value);
      const binWidth=(xRange[1]-xRange[0])/n,counts=Array(n).fill(0);
      plotRows.forEach(row=>counts[Math.min(n-1,Math.max(0,Math.floor((row[xKey]-xRange[0])/binWidth)))]++);
      const top=Math.max(...counts,1);axes(xRange,[0,top],labels[xKey],'Count','Histogram');
      counts.forEach((count,i)=>{
        const left=x(xRange[0]+i*binWidth),width=frame.width/n,height=count/top*frame.height,y=frame.top+frame.height-height;
        ctx.fillStyle='#187b74';ctx.fillRect(left+1,y,Math.max(1,width-2),height);
        bars.push({x:left,y,width,height,count,lo:xRange[0]+i*binWidth,hi:xRange[0]+(i+1)*binWidth});
      });
      lastPlot={type:view,count:plotRows.length,counts,xKey};
    }else{
      const yRange=extent(plotRows.map(row=>row[yKey])),y=v=>frame.top+frame.height-(v-yRange[0])/(yRange[1]-yRange[0])*frame.height;
      axes(xRange,yRange,labels[xKey],labels[yKey],'Scatter plot');
      for(const row of plotRows){const px=x(row[xKey]),py=y(row[yKey]);ctx.beginPath();ctx.arc(px,py,6,0,2*Math.PI);ctx.fillStyle='#187b74';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1;ctx.stroke();points.push({x:px,y:py,row})}
      lastPlot={type:view,count:plotRows.length,xKey,yKey};
    }
  }
  function switchView(next){
    view=next;
    document.querySelectorAll('[data-detail-view]').forEach(button=>{const active=button.dataset.detailView===view;button.classList.toggle('active',active);button.setAttribute('aria-selected',String(active))});
    $('detailsTableView').classList.toggle('hidden',view!=='table');$('detailsPlotView').classList.toggle('hidden',view==='table');
    $('detailsPlotView').setAttribute('aria-labelledby',view==='scatter'?'detailsScatterTab':'detailsHistogramTab');
    $('detailsYAxisLabel').classList.toggle('hidden',view!=='scatter');$('detailsBinsLabel').classList.toggle('hidden',view!=='histogram');$('downloadPlot').classList.toggle('hidden',view==='table');
    update();
  }
  for(const select of [$('detailsXAxis'),$('detailsYAxis')])metrics.forEach(([key,label])=>select.add(new Option(label,key)));
  $('detailsXAxis').value='equivalent_diameter_um';$('detailsYAxis').value='roundness';
  for(const id of ['detailsXAxis','detailsYAxis','detailsBins'])$(id).onchange=drawPlot;
  document.querySelectorAll('[data-detail-view]').forEach(button=>button.onclick=()=>switchView(button.dataset.detailView));
  $('detailsTargetSearch').oninput=renderTargets;
  $('detailsSelectAll').onclick=()=>{selectedIds=new Set((data?.candidates??[]).map(row=>row.candidate_id));allSelected=true;renderTargets();update()};
  $('detailsSelectNone').onclick=()=>{selectedIds.clear();allSelected=false;renderTargets();update()};
  window.renderPoreDetails=next=>{
    if(data?.dataset!==next.dataset){allSelected=true;$('detailsTargetSearch').value='';}
    data=next;const ids=new Set(data.candidates.map(row=>row.candidate_id));
    selectedIds=allSelected?ids:new Set([...selectedIds].filter(id=>ids.has(id)));
    renderTargets();update();
  };
  $('openPoreDetails').onclick=()=>{if(!state)return;if(!data||data.dataset!==state.dataset)window.renderPoreDetails(state);if(!dialog.open)dialog.show();update()};
  window.closePoreDetails=()=>{if(dialog.open)dialog.close();$('detailsTargetPicker').open=false};
  $('closePoreDetails').onclick=window.closePoreDetails;
  dialog.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();window.closePoreDetails();$('openPoreDetails').focus()}});
  document.addEventListener('pointerdown',event=>{if(!$('detailsTargetPicker').contains(event.target))$('detailsTargetPicker').open=false});
  let drag=null;const handle=$('poreDetailsHandle');
  handle.onpointerdown=event=>{if(event.target.closest('button'))return;const rect=dialog.getBoundingClientRect();drag={dx:event.clientX-rect.left,dy:event.clientY-rect.top};handle.setPointerCapture(event.pointerId)};
  handle.onpointermove=event=>{if(!drag)return;const left=Math.max(0,Math.min(innerWidth-dialog.offsetWidth,event.clientX-drag.dx)),top=Math.max(0,Math.min(innerHeight-50,event.clientY-drag.dy));dialog.style.left=left+'px';dialog.style.top=top+'px';dialog.style.right='auto';};
  handle.onpointerup=handle.onpointercancel=()=>{drag=null};
  for(const [corner,label] of [['nw','Top left'],['ne','Top right'],['sw','Bottom left'],['se','Bottom right'],['n','Top'],['s','Bottom'],['w','Left'],['e','Right']]){
    const grip=document.createElement('span');grip.className=corner.length===2?'details-resize-corner':'details-resize-edge';grip.dataset.corner=corner;grip.title=label+' — drag to resize';dialog.append(grip);
    let resizing=null;
    grip.onpointerdown=event=>{
      if(event.button!==0)return;
      event.preventDefault();event.stopPropagation();
      resizing={rect:dialog.getBoundingClientRect(),x:event.clientX,y:event.clientY,pointer:event.pointerId};
      grip.setPointerCapture(event.pointerId);
    };
    grip.onpointermove=event=>{
      if(!resizing||event.pointerId!==resizing.pointer)return;
      const {rect,x,y}=resizing,dx=event.clientX-x,dy=event.clientY-y;
      const west=corner.includes('w'),north=corner.includes('n');
      const maxWidth=Math.min(innerWidth-16,west?rect.right:innerWidth-rect.left),maxHeight=Math.min(innerHeight-16,north?rect.bottom:innerHeight-rect.top);
      const width=corner.includes('w')||corner.includes('e')?Math.max(Math.min(340,maxWidth),Math.min(maxWidth,rect.width+(west?-dx:dx))):rect.width;
      const height=corner.includes('n')||corner.includes('s')?Math.max(Math.min(300,maxHeight),Math.min(maxHeight,rect.height+(north?-dy:dy))):rect.height;
      Object.assign(dialog.style,{left:(west?rect.right-width:rect.left)+'px',top:(north?rect.bottom-height:rect.top)+'px',right:'auto',width:width+'px',height:height+'px'});
    };
    grip.onpointerup=grip.onpointercancel=event=>{if(resizing?.pointer!==event.pointerId)return;resizing=null;if(grip.hasPointerCapture(event.pointerId))grip.releasePointerCapture(event.pointerId)};
    grip.onlostpointercapture=()=>{resizing=null};
  }
  window.addEventListener('resize',()=>{if(!dialog.open)return;const rect=dialog.getBoundingClientRect();dialog.style.left=Math.max(0,Math.min(innerWidth-dialog.offsetWidth,rect.left))+'px';dialog.style.top=Math.max(0,Math.min(innerHeight-50,rect.top))+'px';dialog.style.right='auto'});
  const plotPoint=event=>{const r=plot.getBoundingClientRect();return {x:(event.clientX-r.left)*plot.width/r.width,y:(event.clientY-r.top)*plot.height/r.height}};
  function nearest(point){return points.map(item=>({...item,d:Math.hypot(point.x-item.x,point.y-item.y)})).filter(item=>item.d<16).sort((a,b)=>a.d-b.d)[0]}
  plot.onpointermove=event=>{const point=plotPoint(event);if(view==='scatter'){const match=nearest(point);plot.title=match?'ID '+match.row.candidate_id+' · '+labels[$('detailsXAxis').value]+': '+format(match.row[$('detailsXAxis').value])+' · '+labels[$('detailsYAxis').value]+': '+format(match.row[$('detailsYAxis').value]):'';}else{const match=bars.find(bar=>point.x>=bar.x&&point.x<bar.x+bar.width);plot.title=match?format(match.lo)+' ~ '+format(match.hi)+' · '+match.count+'':''}};
  plot.onclick=event=>{if(view!=='scatter')return;const match=nearest(plotPoint(event));if(match){window.selectPoreFromDetails?.(match.row.candidate_id);drawPlot()}};
  function download(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
  $('downloadDetails').onclick=()=>{
    const columns=tableMetrics,rows=selectedRows(),lines=[['ID',...columns.map(([,label])=>label)]];
    rows.forEach(row=>lines.push([row.candidate_id,...columns.map(([key])=>valid(row[key])?row[key]:'')]));
    for(const name of ['min','max','mean','std'])lines.push([name,...columns.map(([key])=>statistics(rows,key)[name]??'')]);
    const text='\ufeff'+lines.map(row=>row.map(value=>'"'+String(value).replaceAll('"','""')+'"').join(',')).join('\r\n');
    download(new Blob([text],{type:'text/csv;charset=utf-8'}),'pore_details.csv');
  };
  $('downloadPlot').onclick=()=>work('Exporting plot…',async()=>{
    const kind=view,bins=$('detailsBins').value;
    try{
      const result=await api('details-plot',{...payload(),candidate_ids:selectedRows().map(row=>row.candidate_id),kind,x_key:$('detailsXAxis').value,y_key:$('detailsYAxis').value,bins:bins==='auto'?'auto':Number(bins)});
      download(await (await fetch(result.image)).blob(),kind+'.png');
      $('detailsPlotInfo').textContent=result.count+' pores exported.';
    }catch(error){$('detailsPlotInfo').textContent=error.message;throw error;}
  });
  // Expose the plotted counts for reproducible export/validation without modifying masks.
  window.poreDetailsPlotData=()=>lastPlot;
})();
