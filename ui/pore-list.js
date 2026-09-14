(() => {
  let selected=new Set(),currentDataset=null,generation=0,mask=null,tintedColor=null;
  let pickEpoch=0,pickQueue=Promise.resolve(),pendingPicks=0;
  const tinted=document.createElement('canvas');
  function contrastColor(){
    const base=[1,3,5].map(i=>parseInt($('poreColor').value.slice(i,i+2),16));
    const distance=color=>[1,3,5].reduce((sum,i,j)=>sum+(parseInt(color.slice(i,i+2),16)-base[j])**2,0);
    return ['#ff00ff','#00ffff','#ffff00'].sort((a,b)=>distance(b)-distance(a))[0];
  }
  function selectionRows(){
    document.querySelectorAll('#poreListRows tr[data-id]').forEach(row=>{
      const active=selected.has(Number(row.dataset.id));
      row.classList.toggle('selected',active);row.setAttribute('aria-selected',String(active));
    });
    $('selectedPoreCount').textContent=selected.size+' selected';
  }
  async function renderSelection(centerId=null){
    const version=++generation;mask=null;tintedColor=null;selectionRows();draw();controls();
    if(!state||!selected.size){$('poreSelectionInfo').textContent='No selection';return}
    const dataset=state.dataset,revision=state.revision,ids=[...selected];
    $('poreSelectionInfo').textContent=ids.length+' pores selected';
    if(ids.length===1){
      const row=state.candidates.find(c=>c.candidate_id===ids[0]);
      if(row)$('poreSelectionInfo').textContent=ids[0]+' · '+row.area_um2.toFixed(2)+' µm² · Diameter '+row.equivalent_diameter_um.toFixed(2)+' µm · Roundness '+(row.roundness==null?'—':row.roundness.toFixed(3));
    }
    try{
      const result=await api('pore-highlight',{dataset,revision,candidate_ids:ids});
      const loaded=await image(result.image);
      if(version!==generation||state?.dataset!==dataset||state?.revision!==revision)return;
      mask=loaded;draw();
      if(centerId!==null){
        const row=state.candidates.find(c=>c.candidate_id===centerId);
        if(row){
          const scale=canvas.getBoundingClientRect().width/canvas.width;
          for(const view of [$('viewport'),$('originalViewport')])view.scrollTo(Math.max(0,row.centroid_x_pixels*scale-view.clientWidth/2),Math.max(0,row.centroid_y_pixels*scale-view.clientHeight/2));
        }
      }
    }catch(error){if(version===generation)setStatus(error.message,true)}
  }
  function clearSelection(){pickEpoch++;selected.clear();renderSelection()}
  function selectPore(id,toggle=false,center=true){
    if(toggle){if(selected.has(id))selected.delete(id);else selected.add(id)}
    else selected=new Set([id]);
    return renderSelection(center?id:null);
  }
  window.drawSelectedPore=context=>{
    if(!mask)return;
    const color=contrastColor();
    if(color!==tintedColor){
      tinted.width=canvas.width;tinted.height=canvas.height;
      const c=tinted.getContext('2d');c.drawImage(mask,0,0);c.globalCompositeOperation='source-in';
      c.fillStyle=color;c.fillRect(0,0,tinted.width,tinted.height);c.globalCompositeOperation='source-over';tintedColor=color;
    }
    $('poreDetailsDialog').style.setProperty('--highlight-color',color);
    context.save();context.globalAlpha=.8;context.drawImage(tinted,0,0);context.restore();
  };
  window.onPoreListState=data=>{
    pickEpoch++;generation++;mask=null;
    const existing=new Set(data.candidates.map(c=>c.candidate_id));
    selected=currentDataset!==data.dataset?new Set():new Set([...selected].filter(id=>existing.has(id)));
    currentDataset=data.dataset;window.renderPoreDetails?.(data);renderSelection();
  };
  window.selectPoreFromDetails=(id,toggle=false)=>{
    if(!busy&&state?.candidates.some(c=>c.candidate_id===id)){pickEpoch++;selectPore(id,toggle)}
  };
  window.getSelectedPores=()=>[...selected].sort((a,b)=>a-b);
  window.getHighlightedPore=()=>selected.values().next().value??null;
  window.isPoreSelectionPending=()=>pendingPicks>0;
  window.pickPoreAtPoint=(point,toggle=false)=>{
    const epoch=pickEpoch,dataset=state.dataset,revision=state.revision;
    pendingPicks++;controls();
    // Keep rapid Ctrl-clicks in input order, even if requests finish at different times.
    pickQueue=pickQueue.then(async()=>{
      if(epoch!==pickEpoch||state?.dataset!==dataset||state?.revision!==revision||mode!=='select'||busy)return;
      const result=await api('pore-at-point',{dataset,revision,point});
      if(epoch!==pickEpoch||state?.dataset!==dataset||state?.revision!==revision||mode!=='select'||busy)return;
      if(result.candidate_id===null){if(!toggle){selected.clear();await renderSelection()}return}
      await selectPore(result.candidate_id,toggle,false);
      if(epoch===pickEpoch&&mode==='select')setStatus(selected.size+' pores selected');
    }).catch(error=>{if(epoch===pickEpoch)setStatus(error.message,true)}).finally(()=>{pendingPicks--;controls()});
    return pickQueue;
  };
  window.clearPoreHighlight=clearSelection;
  $('clearPoreSelection').onclick=clearSelection;
  $('selectBoundaryPores').onclick=()=>{
    if(busy||!state)return;
    if(mode!=='select')document.querySelector('[data-mode="select"]').click();
    pickEpoch++;selected=new Set(state.candidates.filter(c=>c.touches_image_edge).map(c=>c.candidate_id));
    renderSelection();setStatus('Boundary pores: '+selected.size+' selected');
  };
})();
