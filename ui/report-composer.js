/* Report drafts are independent of masks. Measurements refresh only on request. */
(()=>{
  let data=null,options=null,dirty=false,loading=false,previewGeneration=0,dragging=null,pickerTarget=null,pickerOrder=[],pickerKind='image',reportId=null,workspaceReady=false,savedReportId=null,draftVersion=0,autoTimer=null,autoSaving=null;
  const el=id=>document.getElementById(id);
  const status=text=>{el('reportDraftStatus').textContent=text};
  const key=()=>reportId?reportId+':'+(data?.revision??0):(state?state.dataset+':'+state.revision:'');
  const current=()=>data&&data.dataset+':'+data.revision===key();
  const reportPayload=()=>reportId?{report_id:reportId}:(state?payload():{});
  function reportApi(name,args){
    if(!reportId)return api(name,args);
    const actions={'report-content':'refresh','report-save-draft':'save','report-preview':'preview','generate-report':'export','report-conditions-default':'conditions'};
    return api('report-workspace',{...args,report_id:reportId,action:actions[name]});
  }
  function tab(text){
    el('reportTextFields').classList.toggle('hidden',!text);el('reportAssetFields').classList.toggle('hidden',text);
    for(const [id,active] of [['reportTextTab',text],['reportContentTab',!text]]){el(id).classList.toggle('active',active);el(id).setAttribute('aria-selected',String(active))}
  }
  el('reportTextTab').onclick=()=>tab(true);el('reportContentTab').onclick=()=>tab(false);
  function collect(){
    if(!options)return null;
    options.title=el('reportTitle').value;options.sample='';
    options.method=el('reportMethod').value;options.results=el('reportResults').value;
    options.conditions=[...el('reportConditions').children].map(row=>[row.children[0].value,row.children[1].value]);
    return structuredClone(options);
  }
  function changed(){dirty=true;draftVersion++;status('Saving...');el('reportPreviewStatus').textContent='Updating...';scheduleSave()}
  function scheduleSave(){clearTimeout(autoTimer);autoTimer=setTimeout(autoSave,900)}
  async function autoSave(){
    autoTimer=null;if(!dirty||!data)return;if(busy||loading||autoSaving){scheduleSave();return}
    const version=draftVersion,identity=key();
    const source=data,settings=collect();
    autoSaving=(async()=>{
      if(source.report_id)await api('report-workspace',{action:'save',report_id:source.report_id,options:settings});
      else await api('report-save-draft',{dataset:source.dataset,revision:source.revision,options:settings});
      if(identity===key()&&version===draftVersion){dirty=false;status('Saved');await preview()}
    })();
    try{await autoSaving}catch(error){status('Could not update: '+error.message)}finally{autoSaving=null;if(dirty&&version!==draftVersion)scheduleSave()}
  }

  function conditionRow(pair){
    const row=document.createElement('div');row.className='report-condition';
    for(let i=0;i<2;i++){const input=document.createElement('input');input.value=pair[i];input.maxLength=500;input.placeholder=i?'Enter value':'Condition';input.setAttribute('aria-label',i?'Condition value':'Condition name');input.oninput=changed;row.append(input)}
    const remove=document.createElement('button');remove.textContent='×';remove.title='Remove row';remove.onclick=()=>{row.remove();changed()};row.append(remove);el('reportConditions').append(row);
  }
  const titles={original:'SEM Original',segmentation:'Pore Segmentation',comparison:'Original / Segmentation',hist_diameter:'Equivalent Diameter Histogram',hist_length:'Length Histogram',hist_width:'Width Histogram',hist_area:'Area Histogram',hist_aspect:'Aspect Ratio Histogram',statistics:'Pore Size Statistics'};
  const assetTitle=a=>(a.source_name?a.source_name+' / ':'')+(titles[a.base_id||a.id]||a.title);
  function assets(){return [...data.assets,...options.extra_images]}
  function panelRows(item){
    const ids=item.image_ids??[];
    if(!Array.isArray(item.rows)){
      const count=item.columns||1;item.rows=[];
      for(let i=0;i<ids.length;i+=count)item.rows.push(ids.slice(i,i+count));
    }
    const seen=new Set();item.rows=item.rows.map(row=>row.filter(id=>ids.includes(id)&&!seen.has(id)&&(seen.add(id),true))).filter(row=>row.length);
    for(const id of ids)if(!seen.has(id))item.rows.push([id]);
    item.image_ids=item.rows.flat();return item.rows;
  }
  function clearPanelDrop(){document.querySelectorAll('[data-drop-edge]').forEach(n=>delete n.dataset.dropEdge)}
  function dropEdge(event,box){const r=box.getBoundingClientRect(),x=(event.clientX-r.left)/r.width-.5,y=(event.clientY-r.top)/r.height-.5;return Math.abs(x)>Math.abs(y)?(x<0?'left':'right'):(y<0?'top':'bottom')}
  function movePanel(item,id,target,edge){
    if(id===target)return;
    const rows=panelRows(item).map(row=>row.filter(k=>k!==id)).filter(row=>row.length);
    if(target===null)rows.push([id]);
    else{const r=rows.findIndex(row=>row.includes(target));if(r<0)return;
      if(edge==='left'||edge==='right')rows[r].splice(rows[r].indexOf(target)+(edge==='right'?1:0),0,id);
      else rows.splice(r+(edge==='bottom'?1:0),0,[id]);
    }
    item.rows=rows;item.image_ids=rows.flat();dragging=null;document.body.classList.remove('report-panel-dragging');clearPanelDrop();changed();renderCards();
  }
  function renderCards(){
    const container=el('reportContentCards');container.replaceChildren();
    if(!options.items.length){const hint=document.createElement('p');hint.className='hint';hint.textContent='Choose Images to start, or add a figure or table.';container.append(hint)}
    const catalog=new Map(assets().map(a=>[a.id,a]));
    let figureNumber=0,tableNumber=0;
    options.items.forEach((item,index)=>{
      const asset=catalog.get(item.id)||{kind:'image',image:true};
      const card=document.createElement('article');card.className='report-card';card.dataset.id=item.id;
      const header=document.createElement('div');header.className='row';
      const handle=document.createElement('button');handle.className='report-drag-handle';handle.textContent='☰';handle.title='Drag to reorder content';handle.draggable=true;
      handle.ondragstart=e=>{if(busy){e.preventDefault();return}dragging={kind:'card',id:item.id};e.dataTransfer.setData('text/plain',item.id);e.dataTransfer.effectAllowed='move';card.classList.add('dragging')};
      handle.ondragend=()=>{dragging=null;document.querySelectorAll('.drag-over,.dragging').forEach(n=>n.classList.remove('drag-over','dragging'))};
      card.ondragover=e=>{if(dragging?.kind==='card'&&dragging.id!==item.id){e.preventDefault();e.dataTransfer.dropEffect='move';card.classList.add('drag-over')}};
      card.ondragleave=e=>{if(!card.contains(e.relatedTarget))card.classList.remove('drag-over')};
      card.ondrop=e=>{if(dragging?.kind!=='card')return;e.preventDefault();e.stopPropagation();const from=options.items.findIndex(i=>i.id===dragging.id);if(from<0||dragging.id===item.id)return;const below=e.clientY>card.getBoundingClientRect().top+card.offsetHeight/2;const [moved]=options.items.splice(from,1);const to=options.items.indexOf(item)+(below?1:0);options.items.splice(to,0,moved);dragging=null;changed();renderCards()};
      const label=document.createElement('label');label.className='row';
      const check=document.createElement('input');check.type='checkbox';check.checked=item.selected;check.onchange=()=>{item.selected=check.checked;changed();renderCards()};
      const title=document.createElement('strong');const number=asset.kind==='table'?'Table '+(++tableNumber):'Figure '+(++figureNumber);title.textContent=number;label.append(check,title);header.append(handle,label);card.append(header);
      if(asset.image){
        item.image_ids??=[];panelRows(item);
        const panels=document.createElement('div');panels.className='report-panels';
        item.rows.forEach(rowIds=>{
        const row=document.createElement('div');row.className='report-panel-row';row.style.gridTemplateColumns=`repeat(${rowIds.length},minmax(0,1fr))`;panels.append(row);
        rowIds.forEach(id=>{const panelIndex=item.image_ids.indexOf(id);
          const panel=catalog.get(id),box=document.createElement('div');box.className='report-panel-image';box.dataset.assetId=id;
          const button=document.createElement('button');button.className='report-thumbnail';button.title='Enlarge '+assetTitle(panel);
          const image=document.createElement('img');image.src=panel.image;image.alt=assetTitle(panel);button.append(image);
          button.onclick=()=>{el('reportImageLarge').src=panel.image;el('reportImageDialog').showModal()};box.append(button);
          const name=document.createElement('div');name.className='report-image-name';name.textContent=(panelIndex+1)+'. '+assetTitle(panel);box.append(name);
          box.draggable=true;box.ondragstart=e=>{document.body.classList.add('report-panel-dragging');e.stopPropagation();if(busy){e.preventDefault();return}dragging={kind:'panel',figure:item.id,id};e.dataTransfer.setData('text/plain',id);e.dataTransfer.effectAllowed='move';box.classList.add('dragging')};
          box.ondragend=()=>{document.body.classList.remove('report-panel-dragging');clearPanelDrop();dragging=null;document.querySelectorAll('.drag-over,.dragging').forEach(n=>n.classList.remove('drag-over','dragging'))};
          box.ondragover=e=>{if(dragging?.kind==='panel'&&dragging.figure===item.id&&dragging.id!==id){e.preventDefault();e.stopPropagation();clearPanelDrop();const edge=dropEdge(e,box);(edge==='top'||edge==='bottom'?row:box).dataset.dropEdge=edge;e.dataTransfer.dropEffect='move'}};
          box.ondragleave=e=>{if(!box.contains(e.relatedTarget))clearPanelDrop()};
          box.ondrop=e=>{if(dragging?.kind!=='panel'||dragging.figure!==item.id)return;e.preventDefault();e.stopPropagation();movePanel(item,dragging.id,id,dropEdge(e,box))};
          image.draggable=false;
          {const remove=document.createElement('button');remove.textContent='\u00d7';remove.className='report-remove-panel';remove.title='Remove image';remove.setAttribute('aria-label','Remove image');remove.onclick=()=>{item.image_ids.splice(panelIndex,1);changed();renderCards()};box.append(remove)}
          row.append(box);
        });});
        const end=document.createElement('div');end.className='report-new-row';end.textContent='Drop here to add a row';
        end.ondragover=e=>{if(dragging?.kind==='panel'&&dragging.figure===item.id){e.preventDefault();e.stopPropagation();clearPanelDrop();end.dataset.dropEdge='bottom'}};
        end.ondragleave=clearPanelDrop;
        end.ondrop=e=>{if(dragging?.kind!=='panel'||dragging.figure!==item.id)return;e.preventDefault();e.stopPropagation();movePanel(item,dragging.id,null,'bottom')};
        if(item.image_ids.length)panels.append(end);
        card.append(panels);
        if(!item.image_ids.length){const hint=document.createElement('p');hint.className='hint';hint.textContent='No contents yet. Choose Add Contents.';card.append(hint)}
        const add=document.createElement('button');add.textContent='Add Contents';add.onclick=()=>openContents(item);card.append(add);
      }else{
        const table=document.createElement('table');
        [['Statistic','Diameter (µm)','Length (µm)','Width (µm)','Aspect ratio'],...asset.rows.map((r,i)=>[['Minimum','Maximum','Median','Mean','Std. deviation'][i],...r.slice(1)])].forEach((values,i)=>{const row=document.createElement('tr');for(const v of values){const cell=document.createElement(i?'td':'th');cell.textContent=v;row.append(cell)}table.append(row)});card.append(table);
      }
      const caption=document.createElement('input');caption.value=item.caption;caption.maxLength=500;caption.setAttribute('aria-label','Caption');caption.placeholder=asset.kind==='table'?'Table caption':'Figure caption';caption.oninput=()=>{item.caption=caption.value;changed()};card.append(caption);
      const controls=document.createElement('div');controls.className='row';
      for(const [symbol,delta] of [['↑',-1],['↓',1]]){
        const button=document.createElement('button');button.textContent=symbol;button.title=delta<0?'Move up':'Move down';button.disabled=index+delta<0||index+delta>=options.items.length;
        button.dataset.reportOrderDisabled=String(button.disabled);
        button.onclick=()=>{if(index+delta<0||index+delta>=options.items.length)return;[options.items[index],options.items[index+delta]]=[options.items[index+delta],options.items[index]];changed();renderCards()};controls.append(button);
      }
      const removeFigure=document.createElement('button');removeFigure.textContent='Remove '+(asset.kind==='table'?'Table':'Figure');removeFigure.onclick=()=>{options.items.splice(index,1);changed();renderCards()};controls.append(removeFigure);const menu=document.createElement('details');menu.className='report-item-menu';const menuTitle=document.createElement('summary');menuTitle.textContent='Options';menu.append(menuTitle,controls);header.append(menu);container.append(card);
    });
  }
  function newFigure(){return {id:'figure_'+Date.now().toString(36)+'_'+Math.random().toString(36).slice(2,7),selected:true,caption:'',image_ids:[],columns:1,rows:[]}}
  function migrateItems(){
    options.items=options.items.filter(i=>i.selected||i.id.startsWith('figure_')).map(i=>{
      if(assets().find(a=>a.id===i.id)?.kind==='table')return i;
      const migrated={...i,id:i.id.startsWith('figure_')?i.id:newFigure().id,image_ids:i.image_ids??[i.id]};panelRows(migrated);return migrated;
    });
  }
  function openContents(item,kind='image'){
    pickerKind=kind;pickerTarget=item;pickerOrder=kind==='table'?[]:[...item.image_ids];const container=el('reportContentsChoices');container.replaceChildren();
    el('reportContentsTitle').textContent=kind==='table'?'Add Tables':'Add Contents - Figure '+(options.items.filter(i=>i.id.startsWith('figure_')).indexOf(item)+1);
    const filter=el('reportContentSource');filter.replaceChildren(new Option('All images',''));filter.classList.toggle('hidden',!reportId);for(const source of data.sources??[])filter.add(new Option(source.name,source.dataset));
    for(const asset of assets().filter(a=>kind==='table'?a.kind==='table':a.image)){
      const label=document.createElement('label');label.className='report-content-choice';label.dataset.source=asset.source_dataset||'';
      const check=document.createElement('input');check.type='checkbox';check.value=asset.id;check.checked=kind!=='table'&&item.image_ids.includes(asset.id);check.onchange=()=>{pickerOrder=pickerOrder.filter(id=>id!==asset.id);if(check.checked)pickerOrder.push(asset.id)};
      const img=document.createElement('img');img.src=asset.image;img.alt=assetTitle(asset);
      const name=document.createElement('span');name.textContent=assetTitle(asset);label.append(check);if(asset.image)label.append(img);label.append(name);container.append(label);
    }
    el('reportContentsError').textContent='';el('reportContentsDialog').showModal();
  }
  el('reportContentSource').onchange=()=>{for(const choice of el('reportContentsChoices').children)choice.classList.toggle('hidden',!!el('reportContentSource').value&&choice.dataset.source!==el('reportContentSource').value)};
  el('selectAllReportAnalyses').onclick=()=>el('reportAnalysesChoices').querySelectorAll('input').forEach(c=>c.checked=true);
  el('clearReportAnalyses').onclick=()=>el('reportAnalysesChoices').querySelectorAll('input').forEach(c=>c.checked=false);
  el('cancelReportContents').onclick=()=>el('reportContentsDialog').close();
  el('applyReportContents').onclick=()=>{
    const ids=[...pickerOrder];
    if(pickerKind==='table'){
      for(const id of ids)if(!options.items.some(i=>i.id===id)){const asset=assets().find(a=>a.id===id);options.items.push({id,selected:true,caption:(asset.source_name?asset.source_name+' / ':'')+'Pore size statistics'});}
      changed();renderCards();el('reportContentsDialog').close();return;
    }
    if(ids.length>8){el('reportContentsError').textContent='Choose up to 8 contents.';return}
    pickerTarget.image_ids=ids;
    panelRows(pickerTarget);changed();renderCards();el('reportContentsDialog').close();
  };
  el('addReportFigure').onclick=()=>{if(options.items.length>=100)return;options.items.push(newFigure());changed();renderCards()};
  el('addReportTable').onclick=()=>{if(reportId){openContents(null,'table');return}if(options.items.some(i=>i.id==='statistics'))return;options.items.push({id:'statistics',selected:true,caption:'Pore size statistics'});changed();renderCards()};
  function render(){
    el('analysisFrame').classList.add('hidden');el('analysisComparison').closest('section').classList.add('hidden');el('analysisDownloads').replaceChildren();el('generateReport').textContent='Save PDF + HWPX';
    const sources=reportId?data.sources:[{name:data.name,dataset:data.dataset}];
    el('reportSourcesStatus').textContent=sources.length?`${sources.length} image${sources.length===1?'':'s'} selected`:'Select images for your report.';
    el('reportSourceChips').replaceChildren();
    for(const source of sources){
      const chip=document.createElement('span');chip.className='report-source-chip';chip.title=source.name;
      const asset=data.assets.find(a=>(!reportId||a.source_dataset===source.dataset)&&(a.base_id||a.id)==='original');
      if(asset){const image=document.createElement('img');image.src=asset.image;image.alt='';chip.append(image)}
      const name=document.createElement('span');name.textContent=source.name;chip.append(name);el('reportSourceChips').append(chip);
    }
    el('analysisTitle').textContent='Report';el('analysisSubtitle').textContent='';
    el('reportTitle').value=options.title;el('reportMethod').value=options.method;el('reportResults').value=options.results;
    el('reportConditions').replaceChildren();options.conditions.forEach(conditionRow);renderCards();
    el('reportComposeBody').classList.remove('hidden');
    el('reportSummaryStatus').textContent=options.summary_revision!==data.revision?'Pores changed since this text was drafted. Review the text or regenerate the draft.':'Draft based on measurements. Review and edit as needed.';
  }
  async function preview(){
    if(!current())throw new Error('Load / Update Content for the current pore revision first.');
    const generation=++previewGeneration;const requestKey=key();
    el('reportPreviewStatus').textContent='Preparing preview…';
    const result=await reportApi('report-preview',{...reportPayload(),options:collect()});
    if(generation!==previewGeneration||requestKey!==key())return;
    el('composedReportPreview').srcdoc=result.html;el('reportPreviewStatus').textContent='';
  }
  async function load(){
    if((!state&&!reportId)||loading)return;
    loading=true;status('Loading report content…');
    try{
      // Save edits before switching datasets or refreshing a changed revision.
      await saveCurrent();
      const requestKey=key();const result=await reportApi('report-content',reportPayload());if(requestKey!==key())return;
      data=result;options=result.options;migrateItems();dirty=false;render();status('Saved');await preview();
    }finally{loading=false}
  }
  window.openReportComposer=()=>work('Loading report...',async()=>{await initWorkspaces();if(!current())await load()});
  el('loadReportContent').onclick=()=>work('Updating report content…',load);
  for(const id of ['reportTitle','reportMethod','reportResults'])el(id).oninput=changed;
  el('addReportCondition').onclick=()=>{if(el('reportConditions').children.length>=20)return;conditionRow(['','']);changed()};
  el('refreshReportSummary').onclick=()=>{
    if(!current()){status('Update content before regenerating the draft.');return}
    if(el('reportResults').value!==data.summary&&!confirm('Replace the current results text with a new measurement-based draft?'))return;
    el('reportResults').value=data.summary;options.summary_revision=data.revision;el('reportSummaryStatus').textContent='Draft regenerated from current measurements.';changed();
  };
  el('refreshReportPreview').onclick=()=>work('Updating report preview…',preview);
  el('saveReportDraft').onclick=()=>work('Saving draft...',async()=>{await saveCurrent();status('Saved');el('reportMore').open=false});
  el('closeReportImage').onclick=()=>el('reportImageDialog').close();
  el('setReportConditionsDefault').onclick=()=>work('Saving default conditions...',async()=>{
    await reportApi('report-conditions-default',{...reportPayload(),conditions:collect().conditions});
    el('reportConditionsDefaultStatus').textContent='Default saved. New analyses will use these conditions.';
  });
  el('generateReport').onclick=()=>work('Saving report…',async()=>{
    await saveCurrent();if(!current())await load();if(!current())throw new Error('Load report content first.');
    const directory=el('exportDirectory').value.trim();if(!directory)throw new Error('Choose an export folder.');
    el('reportGenerationStatus').textContent='Saving selected content…';
    try{
      const result=await reportApi('generate-report',{...reportPayload(),export_directory:directory,report_options:collect()});
      dirty=false;status('Draft saved');el('exportResult').textContent='Exported to: '+result.exported_folder;el('exportResult').dataset.revisionKey=key();
      el('reportGenerationStatus').textContent='PDF and HWPX saved in the destination folder.';await preview();
      try{localStorage.setItem('poreExportDirectory',directory)}catch{}
    }catch(e){el('reportGenerationStatus').textContent='Export failed: '+e.message;throw e}
  });
  const accepted=window.onEditorAccepted;
  window.onEditorAccepted=next=>{
    accepted?.(next);if(reportId){el('analysisTitle').textContent='Report';el('analysisSubtitle').textContent='';}
    // The composer has its own preview; legacy measurement reports stay available on disk.
    el('analysisFrame').classList.add('hidden');el('analysisComparison').closest('section').classList.add('hidden');el('analysisDownloads').replaceChildren();
    el('generateReport').textContent='Save PDF + HWPX';
    if(data&&(!reportId?(data.dataset!==next.dataset||data.revision!==next.revision):data.sources.some(s=>s.dataset===next.dataset&&s.revision!==next.revision))){
      status('Pores changed. Update report content.');el('reportPreviewStatus').textContent='Preview is from an earlier analysis revision.';
    }
  };
  async function saveCurrent(){
    clearTimeout(autoTimer);if(autoSaving)try{await autoSaving}catch{}
    if(data&&dirty){
      if(data.report_id)await api('report-workspace',{action:'save',report_id:data.report_id,options:collect()});
      else await api('report-save-draft',{dataset:data.dataset,revision:data.revision,options:collect()});
      dirty=false;
    }
  }
  async function reportList(){
    const result=await api('report-workspace',{action:'list'});const select=el('reportWorkspaceSelect');select.replaceChildren(new Option('Current image report',''));
    for(const r of result.reports)select.add(new Option(r.title+' / '+r.count+' images / '+r.updated.replace('T',' '),r.id));select.value=reportId||'';
    savedReportId=result.active;return result.reports;
  }
  async function useWorkspace(result){
    el('reportMore').open=false;
    reportId=result.report_id;data=result;options=result.options;migrateItems();dirty=false;render();status('Saved');
    try{localStorage.setItem('poreActiveReport',reportId)}catch{}await reportList();await preview();
  }
  async function initWorkspaces(){
    if(workspaceReady)return;workspaceReady=true;const reports=await reportList();let active=savedReportId||'';if(!active)try{active=localStorage.getItem('poreActiveReport')||''}catch{}
    if(reports.some(r=>r.id===active)){await useWorkspace(await api('report-workspace',{action:'load',report_id:active}));return}
    if(!state){await useWorkspace(await api('report-workspace',{action:'create',datasets:[]}));}
  }
  el('newReportWorkspace').onclick=()=>work('Creating report...',async()=>{await saveCurrent();await useWorkspace(await api('report-workspace',{action:'create',datasets:[]}))});
  el('reportWorkspaceSelect').onchange=()=>work('Opening report...',async()=>{
    const id=el('reportWorkspaceSelect').value;await saveCurrent();
    if(id)await useWorkspace(await api('report-workspace',{action:'load',report_id:id}));
    else{reportId=null;data=null;try{localStorage.removeItem('poreActiveReport')}catch{}if(state)await load();else{el('reportComposeBody').classList.add('hidden');status('Open an image, or choose New Report.')}}
  });
  el('addReportAnalyses').onclick=()=>work('Loading saved analyses...',async()=>{
    const result=await fetch('/api/images').then(r=>r.json());const container=el('reportAnalysesChoices');container.replaceChildren();
    const selected=new Set(reportId?data.sources.map(s=>s.dataset):(state?[state.dataset]:[]));
    for(const entry of result.images.filter(e=>e.analyses.length)){
      const label=document.createElement('label');label.className='report-analysis-choice';const check=document.createElement('input');check.type='checkbox';
      const image=document.createElement('img');image.src=entry.preview_url;image.alt=entry.name;const title=document.createElement('span');title.textContent=entry.name;
      const select=document.createElement('select');select.setAttribute('aria-label','Analysis for '+entry.name);
      for(const a of entry.analyses)select.add(new Option(a.label+(a.revision?' / Revision '+a.revision:''),a.dataset));
      const included=entry.analyses.find(a=>selected.has(a.dataset));check.checked=!!included;if(included)select.value=included.dataset;
      label.append(check,image,title);const version=document.createElement('details');version.className='report-analysis-version';const versionTitle=document.createElement('summary');versionTitle.textContent='Analysis version';version.append(versionTitle,select);if(entry.analyses.length<2)version.hidden=true;label.append(version);container.append(label);
    }
    el('reportAnalysesError').textContent=container.children.length?'':'No saved analyses yet.';el('reportAnalysesDialog').showModal();
  });
  el('cancelReportAnalyses').onclick=()=>el('reportAnalysesDialog').close();
  el('applyReportAnalyses').onclick=()=>work('Updating report analyses...',async()=>{
    await saveCurrent();
    const datasets=[...el('reportAnalysesChoices').children].filter(row=>row.querySelector('input').checked).map(row=>row.querySelector('select').value);
    if(datasets.length>30){el('reportAnalysesError').textContent='Choose up to 30 analyses.';return}
    if(!reportId){
      const created=await api('report-workspace',{action:'create',datasets:state?[state.dataset]:[],...(data&&state?{import_options:collect()}: {})});reportId=created.report_id;data=created;options=created.options;
    }
    const result=await api('report-workspace',{action:'refresh',report_id:reportId,datasets,options:collect()});
    if(!result.options.items.length){for(const source of result.sources){const asset=result.assets.find(a=>a.source_dataset===source.dataset&&a.base_id==='comparison');if(asset)result.options.items.push({...newFigure(),caption:source.name,image_ids:[asset.id],rows:[[asset.id]]});}await api('report-workspace',{action:'save',report_id:result.report_id,options:result.options});}
    el('reportAnalysesDialog').close();await useWorkspace(result);
  });
  window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});
  const updateControls=window.updateWorkspaceControls;
  window.updateWorkspaceControls=()=>{
    updateControls?.();document.querySelector('[data-panel=analysis]').disabled=busy;
    for(const element of document.querySelectorAll('#reportComposer textarea'))element.disabled=busy;
    for(const button of document.querySelectorAll('[data-report-order-disabled]'))button.disabled=busy||button.dataset.reportOrderDisabled==='true';
  };
})();
