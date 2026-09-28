(() => {
  const dialog=$('poreDetailsDialog'),plot=$('detailsPlot'),ctx=plot.getContext('2d');
  const metrics=window.poreColumnCatalog.filter(item=>!item[3]),labels=Object.fromEntries(metrics.map(([key,label])=>[key,label]));
  const tableColumns=window.createPoreColumnPicker(()=>{sortKey='candidate_id';sortAscending=true;renderTable()});
  const format=value=>value==null||!Number.isFinite(value)?'—':value.toLocaleString(undefined,{maximumFractionDigits:3,minimumFractionDigits:3});
  let data=null,selectedIds=new Set(),allSelected=true,view='table',points=[],bars=[],plotRows=[],lastPlot=null;
  const selectedRows=()=>data?data.candidates.filter(row=>selectedIds.has(row.candidate_id)):[];
  const valid=value=>typeof value==='number'&&Number.isFinite(value);
  let sortKey='candidate_id',sortAscending=true;
  const sortedRows=()=>selectedRows().sort((a,b)=>{
    const av=a[sortKey],bv=b[sortKey],aValid=valid(av)||typeof av==='boolean',bValid=valid(bv)||typeof bv==='boolean';
    // Missing measurements stay last in either direction; ties keep stable IDs.
    if(aValid!==bValid)return aValid?-1:1;
    return (aValid&&av!==bv?(av-bv)*(sortAscending?1:-1):a.candidate_id-b.candidate_id);
  });
  function renderSortHeaders(){
    const row=document.createElement('tr');
    for(const [key,label,help] of tableColumns()){
      const header=document.createElement('th'),button=document.createElement('button'),active=sortKey===key;
      header.scope='col';header.title=help;header.setAttribute('aria-sort',active?(sortAscending?'ascending':'descending'):'none');
      button.type='button';button.className='details-sort';button.dataset.sortKey=key;
      button.textContent=label+' '+(active?(sortAscending?'▲':'▼'):'↕');
      button.setAttribute('aria-label',label+': sort '+(active&&sortAscending?'descending':'ascending'));
      button.onclick=()=>{sortAscending=sortKey===key?!sortAscending:true;sortKey=key;renderTable()};
      header.append(button);row.append(header);
    }
    $('poreTable').querySelector('thead').replaceChildren(row);
  }
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
    renderSortHeaders();
    const fragment=document.createDocumentFragment(),highlight=new Set(window.getSelectedPores?.()??[]);
    for(const candidate of sortedRows()){
      const row=document.createElement('tr');row.dataset.id=candidate.candidate_id;row.tabIndex=0;
      row.classList.toggle('selected',highlight.has(candidate.candidate_id));row.setAttribute('aria-selected',String(highlight.has(candidate.candidate_id)));
      for(const [key,,help,type] of tableColumns()){
        const cell=document.createElement('td');cell.title=help;
        cell.textContent=type==='id'?candidate.candidate_id+(candidate.touches_image_edge?' †':''):type==='boolean'?(candidate[key]==null?'—':candidate[key]?'Yes':'No'):format(candidate[key]);
        row.append(cell);
      }
      row.onclick=event=>window.selectPoreFromDetails?.(candidate.candidate_id,event.ctrlKey);
      row.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();window.selectPoreFromDetails?.(candidate.candidate_id,event.ctrlKey)}};
      fragment.append(row);
    }
    if(!fragment.childNodes.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=tableColumns().length;cell.textContent='No pores selected.';row.append(cell);fragment.append(row)}
    $('poreListRows').replaceChildren(fragment);
    const footer=document.createDocumentFragment(),columns=tableColumns();
    for(const name of ['min','max','mean','std']){
      const row=document.createElement('tr');
      columns.forEach(([key,,help,type],index)=>{
        const cell=document.createElement('td');
        if(index===0){const label=document.createElement('strong');label.className='stat-label';label.textContent=name+' ';cell.append(label)}
        cell.append(document.createTextNode(type?'':format(statistics(selectedRows(),key)[name])));cell.title=help;row.append(cell);
      });footer.append(row);
    }
    $('poreDetailsStats').replaceChildren(footer);
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
    // The table filter controls measurements shown here, not the editor's
    // selection. Hidden rows must never deselect pores in the image.
    renderTable();if(view!=='table')drawPlot();
  }
  function extent(values){
    let lo=Math.min(...values),hi=Math.max(...values);
    if(lo===hi){const d=Math.max(Math.abs(lo)*.05,.5);lo-=d;hi+=d;}
    return [lo,hi];
  }
  function readableAxis(lo,hi,integer=false){
    const raw=(hi-lo)/5,power=10**Math.floor(Math.log10(raw));
    let step=[1,2,5,10].map(v=>v*power).find(v=>v>=raw);
    if(integer)step=Math.max(1,step);
    const start=Math.floor(lo/step)*step,end=Math.ceil(hi/step)*step;
    const digits=Math.max(0,-Math.floor(Math.log10(step)));
    const ticks=Array.from({length:Math.round((end-start)/step)+1},(_,i)=>{
      const value=Number((start+i*step).toPrecision(12));
      const label=value===0?'0':Math.abs(value)>=1e6||Math.abs(value)<1e-3?value.toExponential(2).replace(/\.?0+e/,'e'):value.toLocaleString('en-US',{maximumFractionDigits:Math.min(12,digits)});
      return {value,label};
    });
    return {range:[start,end],ticks};
  }
  const plotWidth=1100,plotHeight=620;
  const frame={left:100,top:68,width:950,height:455};
  function resizePlot(){
    const rect=plot.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
    const width=Math.max(1,Math.round(rect.width*dpr)),height=Math.max(1,Math.round(rect.height*dpr));
    if(plot.width!==width)plot.width=width;
    if(plot.height!==height)plot.height=height;
    ctx.setTransform(width/plotWidth,0,0,height/plotHeight,0,0);
  }
  function axes(xRange,yRange,xLabel,yLabel,title,xTicks=null,yTicks=null){
    ctx.fillStyle='#ffffff';ctx.fillRect(0,0,plotWidth,plotHeight);
    ctx.fillStyle='#334155';ctx.font='bold 22px sans-serif';ctx.textAlign='center';ctx.fillText(title,plotWidth/2,32);
    ctx.font='16px sans-serif';
    const fallback=range=>Array.from({length:6},(_,i)=>{const value=range[0]+i*(range[1]-range[0])/5;return {value,label:Number(value.toPrecision(4)).toString()}});
    for(const tick of xTicks??fallback(xRange)){
      const x=frame.left+(tick.value-xRange[0])/(xRange[1]-xRange[0])*frame.width;
      ctx.strokeStyle='#e2e8f0';ctx.beginPath();ctx.moveTo(x,frame.top);ctx.lineTo(x,frame.top+frame.height);ctx.stroke();
      ctx.fillStyle='#64748b';ctx.textAlign='center';ctx.fillText(tick.label,x,frame.top+frame.height+27);
    }
    for(const tick of yTicks??fallback(yRange)){
      const y=frame.top+frame.height-(tick.value-yRange[0])/(yRange[1]-yRange[0])*frame.height;
      ctx.strokeStyle='#e2e8f0';ctx.beginPath();ctx.moveTo(frame.left,y);ctx.lineTo(frame.left+frame.width,y);ctx.stroke();
      ctx.fillStyle='#64748b';ctx.textAlign='right';ctx.fillText(tick.label,frame.left-12,y+5);
    }
    ctx.strokeStyle='#94a3b8';ctx.strokeRect(frame.left,frame.top,frame.width,frame.height);
    ctx.fillStyle='#334155';ctx.textAlign='center';ctx.fillText(xLabel,frame.left+frame.width/2,plotHeight-26);
    ctx.save();ctx.translate(25,frame.top+frame.height/2);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();
  }
  function drawPlot(){
    resizePlot();
    points=[];bars=[];const xKey=$('detailsXAxis').value,yKey=$('detailsYAxis').value;
    const rows=selectedRows();plotRows=rows.filter(row=>valid(row[xKey])&&(view!=='scatter'||valid(row[yKey])));
    const columns=view==='scatter'?[[xKey,labels[xKey]],[yKey,labels[yKey]]]:[[xKey,labels[xKey]]];
    const header=document.createElement('tr');header.append(document.createElement('th'));
    columns.forEach(([,label])=>{const th=document.createElement('th');th.textContent=label;header.append(th)});
    $('detailsPlotStats').replaceChildren(header,statRows(plotRows,columns));
    $('detailsPlotInfo').textContent='Showing '+plotRows.length+' / Selected: '+rows.length+''+(view==='scatter'?'':'');
    if(!plotRows.length){ctx.fillStyle='white';ctx.fillRect(0,0,plotWidth,plotHeight);ctx.fillStyle='#64748b';ctx.font='22px sans-serif';ctx.textAlign='center';ctx.fillText('No valid values.',plotWidth/2,plotHeight/2);lastPlot={type:view,count:0};return;}
    const xRange=extent(plotRows.map(row=>row[xKey]));
    if(view==='histogram'){
      const targetBins=$('detailsBins').value==='auto'?Math.min(40,Math.max(1,Math.ceil(Math.sqrt(plotRows.length)))):Number($('detailsBins').value);
      const lo=Math.floor(Math.min(...plotRows.map(row=>row[xKey]))),hi=Math.max(...plotRows.map(row=>row[xKey]));
      const binWidth=Math.max(1,Math.ceil((hi-lo)/targetBins)),n=Math.max(1,Math.ceil((hi-lo)/binWidth));
      const edges=Array.from({length:n+1},(_,i)=>lo+i*binWidth),counts=Array(n).fill(0);
      plotRows.forEach(row=>counts[Math.min(n-1,Math.max(0,Math.floor((row[xKey]-lo)/binWidth)))]++);
      const tickEvery=Math.max(1,Math.ceil(n/6));
      const xAxis={range:[edges[0],edges[n]],ticks:edges.filter((_,i)=>i%tickEvery===0||i===n).map(value=>({value,label:value.toLocaleString('en-US')}))},yAxis=readableAxis(0,Math.max(...counts,1)*1.1,true),top=yAxis.range[1];
      const histX=v=>frame.left+(v-xAxis.range[0])/(xAxis.range[1]-xAxis.range[0])*frame.width;
      axes(xAxis.range,yAxis.range,labels[xKey],'Count','Histogram',xAxis.ticks,yAxis.ticks);
      counts.forEach((count,i)=>{
        const left=histX(edges[i]),width=binWidth/(xAxis.range[1]-xAxis.range[0])*frame.width,height=count/top*frame.height,y=frame.top+frame.height-height;
        ctx.fillStyle='#2563eb';ctx.fillRect(left+1,y,Math.max(1,width-2),height);
        bars.push({x:left,y,width,height,count,lo:edges[i],hi:edges[i+1]});
      });
      lastPlot={type:view,count:plotRows.length,counts,edges,xKey,xAxis,yAxis};
    }else{
      const xAxis=readableAxis(...xRange),yAxis=readableAxis(...extent(plotRows.map(row=>row[yKey])));
      const scatterX=v=>frame.left+(v-xAxis.range[0])/(xAxis.range[1]-xAxis.range[0])*frame.width;
      const y=v=>frame.top+frame.height-(v-yAxis.range[0])/(yAxis.range[1]-yAxis.range[0])*frame.height;
      axes(xAxis.range,yAxis.range,labels[xKey],labels[yKey],'Scatter plot',xAxis.ticks,yAxis.ticks);
      for(const row of plotRows){const px=scatterX(row[xKey]),py=y(row[yKey]);ctx.beginPath();ctx.arc(px,py,6,0,2*Math.PI);ctx.fillStyle='#2563eb';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1;ctx.stroke();points.push({x:px,y:py,row})}
      lastPlot={type:view,count:plotRows.length,xKey,yKey,xAxis,yAxis};
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
    $('poreDetailsTitle').textContent='Pore Details';
    if(data?.dataset!==next.dataset){allSelected=true;sortKey='candidate_id';sortAscending=true;$('detailsTargetSearch').value='';}
    const previousIds=new Set(data?.candidates.map(row=>row.candidate_id)??[]);
    data=next;const ids=new Set(data.candidates.map(row=>row.candidate_id));
    selectedIds=allSelected?ids:new Set([...selectedIds].filter(id=>ids.has(id)).concat([...ids].filter(id=>!previousIds.has(id))));
    renderTargets();update();
  };
  window.markPoreDetailsStale=next=>{
    if(data&&(data.dataset!==next.dataset||data.revision!==next.revision))$('poreDetailsTitle').textContent='Pore Details · Click Pore Details to update';
  };
  $('openPoreDetails').onclick=()=>work('Updating pore details…',async()=>{
    if(!state)return;
    window.detailsSnapshot=await measurementSnapshot();
    window.renderPoreDetails(window.detailsSnapshot);
    if(!window.isDetachedDetails&&window.pywebview?.api?.open_details){
      try{await window.pywebview.api.open_details()}catch(error){setStatus(error.message,true)}
      return;
    }
    if(!dialog.open)dialog.show();
  });
  window.closePoreDetails=()=>{if($('poreColumnsDialog').open)$('poreColumnsDialog').close();if(dialog.open)dialog.close();$('detailsTargetPicker').open=false};
  if(window.isDetachedDetails)window.closePoreDetails=()=>window.pywebview.api.close_details();
  $('closePoreDetails').onclick=window.closePoreDetails;
  dialog.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();window.closePoreDetails();$('openPoreDetails').focus()}});
  document.addEventListener('pointerdown',event=>{if(!$('detailsTargetPicker').contains(event.target))$('detailsTargetPicker').open=false});
  if(!window.isDetachedDetails){
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
  }
  const plotPoint=event=>{const r=plot.getBoundingClientRect();return {x:(event.clientX-r.left)*plotWidth/r.width,y:(event.clientY-r.top)*plotHeight/r.height}};
  function nearest(point){return points.map(item=>({...item,d:Math.hypot(point.x-item.x,point.y-item.y)})).filter(item=>item.d<16).sort((a,b)=>a.d-b.d)[0]}
  plot.onpointermove=event=>{const point=plotPoint(event);if(view==='scatter'){const match=nearest(point);plot.title=match?'ID '+match.row.candidate_id+' · '+labels[$('detailsXAxis').value]+': '+format(match.row[$('detailsXAxis').value])+' · '+labels[$('detailsYAxis').value]+': '+format(match.row[$('detailsYAxis').value]):'';}else{const match=bars.find(bar=>point.x>=bar.x&&point.x<bar.x+bar.width);plot.title=match?format(match.lo)+' ~ '+format(match.hi)+' · '+match.count+'':''}};
  plot.onclick=event=>{if(view!=='scatter')return;const match=nearest(plotPoint(event));if(match){window.selectPoreFromDetails?.(match.row.candidate_id);drawPlot()}};
  function download(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
  $('downloadDetails').onclick=()=>{
    const columns=tableColumns(),rows=sortedRows(),lines=[columns.map(([,label])=>label)];
    rows.forEach(row=>lines.push(columns.map(([key,,,type])=>type==='boolean'?(row[key]==null?'':row[key]?'Yes':'No'):valid(row[key])?row[key]:'')));
    lines.push([],['Statistic',...columns.map(([,label])=>label)]);
    for(const name of ['min','max','mean','std'])lines.push([name,...columns.map(([key,,,type])=>type?'':statistics(rows,key)[name]??'')]);
    const text='\ufeff'+lines.map(row=>row.map(value=>'"'+String(value).replaceAll('"','""')+'"').join(',')).join('\r\n');
    download(new Blob([text],{type:'text/csv;charset=utf-8'}),'pore_details.csv');
  };
  $('downloadPlot').onclick=()=>work('Exporting plot…',async()=>{
    const kind=view,bins=$('detailsBins').value;
    try{
      const result=await api('details-plot',{dataset:data.dataset,revision:data.revision,candidate_ids:selectedRows().map(row=>row.candidate_id),kind,x_key:$('detailsXAxis').value,y_key:$('detailsYAxis').value,bins:bins==='auto'?'auto':Number(bins)});
      download(await (await fetch(result.image)).blob(),kind+'.png');
      $('detailsPlotInfo').textContent=result.count+' pores exported.';
    }catch(error){$('detailsPlotInfo').textContent=error.message;throw error;}
  });
  let plotResizeFrame=null;
  function schedulePlotResize(){
    if(plotResizeFrame!==null)return;
    plotResizeFrame=requestAnimationFrame(()=>{plotResizeFrame=null;if(view!=='table'&&plot.getBoundingClientRect().width)drawPlot()});
  }
  new ResizeObserver(schedulePlotResize).observe(plot);
  window.addEventListener('resize',schedulePlotResize);
  // Expose the plotted counts for reproducible export/validation without modifying masks.
  window.poreDetailsPlotData=()=>lastPlot;
})();
