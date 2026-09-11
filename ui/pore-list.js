(() => {
  let selected=null,currentDataset=null,generation=0,mask=null,tintedColor=null;
  const tinted=document.createElement('canvas');
  const sources={automatic:'자동',manual_polygon:'직접 그림',prompted_sam:'SAM 추가',manual_cut:'절단'};
  const format=(value,digits=2)=>Number(value).toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits});
  function contrastColor(){
    const hex=$('poreColor').value,base=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));
    const choices=['#ff00ff','#00ffff','#ffff00'];
    const distance=color=>[1,3,5].reduce((sum,i,j)=>sum+(parseInt(color.slice(i,i+2),16)-base[j])**2,0);
    return choices.sort((a,b)=>distance(b)-distance(a))[0];
  }
  function selectionRows(){
    document.querySelectorAll('#poreListRows tr').forEach(row=>{
      const active=Number(row.dataset.id)===selected;
      row.classList.toggle('selected',active);row.setAttribute('aria-selected',String(active));
    });
  }
  function clearSelection(){
    generation++;selected=null;mask=null;selectionRows();$('poreSelectionInfo').textContent='pore를 선택하세요.';draw();
  }
  async function selectPore(id,center=true){
    selected=id;mask=null;tintedColor=null;selectionRows();draw();
    const version=++generation,dataset=state.dataset,revision=state.revision;
    const row=state.candidates.find(c=>c.candidate_id===id);
    $('poreSelectionInfo').textContent=id+'번 pore 표시 중…';
    try{
      const result=await api('pore-highlight',{dataset,revision,candidate_id:id});
      const loaded=await image(result.image);
      if(version!==generation||state?.dataset!==dataset||state?.revision!==revision)return;
      mask=loaded;
      $('poreSelectionInfo').textContent=id+'번 · '+format(row.area_um2)+' µm² · 직경 '+format(row.equivalent_diameter_um)+' µm · Roundness '+(row.roundness==null?'—':format(row.roundness,3));
      draw();
      if(center){
        const scale=canvas.getBoundingClientRect().width/canvas.width;
        for(const view of [$('viewport'),$('originalViewport')])view.scrollTo(Math.max(0,row.centroid_x_pixels*scale-view.clientWidth/2),Math.max(0,row.centroid_y_pixels*scale-view.clientHeight/2));
      }
    }catch(error){if(version===generation)$('poreSelectionInfo').textContent=error.message;}
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
    generation++;mask=null;
    if(currentDataset!==data.dataset||!data.candidates.some(c=>c.candidate_id===selected))selected=null;
    currentDataset=data.dataset;
    window.renderPoreDetails?.(data);selectionRows();
    if(selected!==null)selectPore(selected,false);else $('poreSelectionInfo').textContent='pore를 선택하세요.';
  };
  window.selectPoreFromDetails=id=>{if(!busy&&state?.candidates.some(c=>c.candidate_id===id))selectPore(id)};
  window.getHighlightedPore=()=>selected;
  window.clearPoreHighlight=clearSelection;
  $('clearPoreSelection').onclick=clearSelection;
})();
