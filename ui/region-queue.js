// Queue drawing prompts; each mask still requires an individual reviewed Apply.
(() => {
  let active=-1,preserve=false,queueDataset=null;
  const panel=document.createElement('div');panel.id='regionQueue';
  const finish=document.createElement('button');finish.id='finishRegion';finish.textContent='Finish Region';finish.title='Finish this region (Enter)';
  const list=document.createElement('div');list.id='regionQueueList';list.setAttribute('aria-label','Queued regions');
  const update=document.createElement('button');update.id='updateRegionPreview';update.textContent='Update Preview';
  panel.append(finish,update,list);$('predict').before(panel);
  function draft(){
    if(!box&&!shape&&polygon.length<3)return null;
    return structuredClone({points,labels,box,shape,polygon,fill_holes:true,target:$('target').value});
  }
  const signature=region=>JSON.stringify(region.input);
  const dirty=region=>!region.token||region.signature!==signature(region);
  function storePoints(){
    const region=queuedRegions[active];
    if(!region||region.status==='Added'||draft()||!points.length)return;
    rememberInput();region.input.points.push(...points);region.input.labels.push(...labels);
    points=[];labels=[];region.status='Pending';
  }
  async function pruneCache(){
    await api('clear-preview-queue',{...payload(),keep_tokens:queuedRegions.filter(r=>r.status!=='Added'&&r.token).map(r=>r.token)});
  }
  async function generate(i){
    const region=queuedRegions[i];region.status='Processing';window.renderRegionQueue();
    try{
      const result=await api(region.kind==='Polygon'?'polygon':'predict',{...region.input,...payload(),queue_preview:true,replace_queue_token:region.token});
      region.token=result.token;region.signature=signature(region);region.status='Ready';region.error=null;
    }catch(error){region.status='Failed';region.error=error.message;throw error}
  }
  update.onclick=()=>work('Updating preview...',async()=>{
    const index=active;if(index<0)return;storePoints();invalidate();await pruneCache();
    await generate(index);await review(index);
  });
  window.canQueueRegion=()=>{
    if(queuedRegions.length>=32&&draft()){setStatus('Maximum 32 regions. Preview or clear the queue.',true);return false}
    return true;
  };
  window.queueCurrentRegion=()=>{
    const input=draft();if(!input)return false;
    if(queuedRegions.length>=32){setStatus('Maximum 32 regions. Preview or clear the queue.',true);return false}
    queuedRegions.push({input,kind:polygon.length>=3?'Polygon':shape?'Ellipse':'Box',status:'Pending',token:null,choice:0});
    queueDataset=state?.dataset;points=[];labels=[];box=null;shape=null;polygon=[];active=-1;
    window.renderRegionQueue();return true;
  };
  window.clearRegionQueue=()=>{if(preserve)return;queuedRegions=[];active=-1;queueDataset=null;window.renderRegionQueue()};
  window.renderRegionQueue=()=>{
    list.replaceChildren();
    finish.classList.toggle('hidden',!['box','ellipse','polygon'].includes(mode));
    const pending=queuedRegions.filter(r=>r.status!=='Added'&&dirty(r)).length;
    $('predict').textContent=queuedRegions.length?'Preview All ('+pending+')':'Preview';
    const current=queuedRegions[active];
    update.classList.toggle('hidden',!current||current.status==='Added'||!['positive','negative','box','ellipse'].includes(mode));
    update.disabled=busy||!current||(!dirty(current)&&!points.length);
    if(current&&(dirty(current)||points.length)&&!draft())$('apply').disabled=true;
    queuedRegions.forEach((region,i)=>{
      const row=document.createElement('div');row.className='queued-region';
      const button=document.createElement('button');button.className='wide';button.classList.toggle('active',i===active);
      button.textContent=(i+1)+'. '+region.kind+' · '+region.status;button.title=region.error||'Review this region';
      button.disabled=busy||region.status==='Added';
      button.onclick=()=>work('Loading region '+(i+1)+'…',()=>review(i));
      const remove=document.createElement('button');remove.textContent='×';remove.title='Remove region';remove.disabled=busy;
      remove.onclick=()=>{rememberInput();queuedRegions.splice(i,1);active=-1;invalidate();window.renderRegionQueue();draw()};
      row.append(button,remove);list.append(row);
    });
  };
  async function review(i){
    const region=queuedRegions[i];if(!region||region.status==='Added')return;
    storePoints();
    if(active>=0&&queuedRegions[active]&&preview)queuedRegions[active].choice=choice;
    invalidate();active=i;
    if(dirty(region)){window.renderRegionQueue();draw();setStatus('Region '+(i+1)+' - Update Preview.');return;}
    let data;
    try{data=await api('queued-preview',{...payload(),queue_token:region.token})}catch(error){region.token=null;region.status='Pending';window.renderRegionQueue();throw error}
    $('target').value=region.input.target;
    await presentPreview(data);
    if(region.choice<data.choices.length)await selectChoice(region.choice);
    window.renderRegionQueue();draw();setStatus('Region '+(i+1)+' · Review and add.');
  }
  finish.onclick=()=>{if(busy||!state)return;rememberInput();if(window.queueCurrentRegion()){invalidate();draw()}else setStatus('Draw a box, ellipse or at least three polygon vertices.',true)};
  document.addEventListener('keydown',event=>{
    if(event.key!=='Enter'||event.repeat||busy||$('editorPanel').classList.contains('hidden')||!['box','ellipse','polygon'].includes(mode))return;
    if(event.target.closest('input,textarea,select,button,dialog'))return;
    event.preventDefault();finish.click();
  });
  window.drawRegionQueue=context=>{
    context.save();context.strokeStyle='#66d9ff';context.fillStyle='#66d9ff';context.lineWidth=2;context.setLineDash([5,4]);context.font='bold 13px Segoe UI';
    queuedRegions.forEach((region,i)=>{
      if(region.status==='Added')return;
      const r=region.input;let x,y;
      if(r.polygon.length>=3){[x,y]=r.polygon[0];context.beginPath();r.polygon.forEach((p,j)=>j?context.lineTo(...p):context.moveTo(...p));context.closePath();context.stroke()}
      else if(r.shape){const s=r.shape;x=s.cx-s.rx;y=s.cy-s.ry;context.beginPath();context.ellipse(s.cx,s.cy,s.rx,s.ry,0,0,2*Math.PI);context.stroke()}
      else{[x,y]=r.box;context.strokeRect(x,y,r.box[2]-x,r.box[3]-y)}
      context.fillText(String(i+1),x+3,Math.max(14,y-4));
    });context.restore();
  };
  const oldControls=controls;
  controls=function(){oldControls();window.renderRegionQueue()};
  const oldPredict=$('predict').onclick;
  $('predict').onclick=()=>{
    if(busy||!state)return;
    if(!queuedRegions.length&&!draft())return oldPredict();
    return work('Preparing queued regions…',async()=>{
      storePoints();
      if(draft()){rememberInput();if(!window.queueCurrentRegion())throw new Error('Maximum 32 regions. Clear some regions first.')}
      const selected=active;
      await pruneCache();invalidate();
      for(let i=0;i<queuedRegions.length;i++){
        const region=queuedRegions[i];if(region.status==='Added'||!dirty(region))continue;
        setStatus('Processing region '+(i+1)+' / '+queuedRegions.length+'...');
        try{await generate(i)}catch(error){region.error=error.message}
      }
      const first=selected>=0&&queuedRegions[selected]?.status==='Ready'?selected:queuedRegions.findIndex(r=>r.status==='Ready');
      if(first>=0)await review(first);else setStatus('No previews available. Check failed regions.',true);
      window.renderRegionQueue();draw();
    });
  };
  const oldApply=$('apply').onclick;
  $('apply').onclick=()=>{
    if(active<0||!queuedRegions[active]?.token)return oldApply();
    return work('Adding region '+(active+1)+'…',async()=>{
      const region=queuedRegions[active],index=active,target=$('target').value;
      const result=await api('apply',{...payload(),token:preview.token,choice,target_id:target?Number(target):null});
      region.status='Added';region.token=null;active=-1;
      preserve=true;try{await accept(result)}finally{preserve=false}
      inputHistory=[];window.renderRegionQueue();draw();
      const next=queuedRegions.findIndex(r=>r.status==='Ready');
      if(next>=0)await review(next);else setStatus('Region '+(index+1)+' added.');
    });
  };
  $('automatePores').onclick=()=>work('Finding additional pores...',async()=>{
    if(!state)return;
    const result=await api('automate-boxes',payload());let added=0;
    // Preserve drawing drafts while each successful addition updates measurements.
    const draftState=structuredClone({points,labels,box,shape,polygon,cutPath,cutPaths,queuedRegions,inputHistory});
    const previousActive=active;
    try{
      for(let i=0;i<result.boxes.length;i++){
        setStatus('Automate '+(i+1)+' / '+result.boxes.length+' - '+added+' added');
        const response=await api('automate-add',{...payload(),box:result.boxes[i].box});
        if(response.added){
          added++;preserve=true;try{await accept(response.state)}finally{preserve=false}
        }
      }
      setStatus('Automate complete: '+added+' pores added.');
    }finally{
      ({points,labels,box,shape,polygon,cutPath,cutPaths,queuedRegions,inputHistory}=draftState);active=previousActive;
      invalidate();window.renderRegionQueue();draw();
    }
  });
  // Returning to an unrelated image cannot reuse its pending prompts.
  const accepted=window.onEditorAccepted;
  window.onEditorAccepted=data=>{if(queueDataset&&queueDataset!==data.dataset){preserve=false;window.clearRegionQueue()}accepted?.(data)};
  window.renderRegionQueue();
})();
