let setupProject=null,setupImage=null,scalePoints=[],measuringScale=false,imageLibrary=[],datasetLabels={};
function beginStopOperation(buttonId,label,notice){
  const operation={buttonId,label,requested:false,stopping:false,cancel:null,
    render(){const button=$(buttonId);button.classList.add('stop-operation');button.textContent=this.stopping?'Stopping…':'Stop';button.disabled=this.stopping;},
    markStopping(){this.requested=true;this.stopping=true;this.render();},
    async stop(){
      if(this.stopping)return;
      this.markStopping();notice('Stopping…');
      if(this.cancel)await this.sendStop();
    },
    async sendStop(){
      try{await this.cancel()}catch(error){
        if(window.activeStopOperation!==this)return;
        this.stopping=false;this.render();notice('Could not stop: '+error.message,true);
      }
    },
    connect(cancel){this.cancel=cancel;if(this.requested)this.sendStop();},
    finish(){if(window.activeStopOperation!==this)return;window.activeStopOperation=null;const button=$(buttonId);button.classList.remove('stop-operation');button.textContent=label;button.disabled=busy;}
  };
  window.activeStopOperation=operation;operation.render();return operation;
}
const setupCanvas=$('setupCanvas'),setupContext=setupCanvas.getContext('2d');
function showPanel(name){
  if(!['load','setup','editor','analysis'].includes(name))return;
  if(['editor','analysis'].includes(name)&&!state)return;
  if(name==='setup'&&!setupProject)return;
  for(const panel of ['load','setup','editor','analysis'])$(panel+'Panel').classList.toggle('hidden',panel!==name);
  document.querySelectorAll('.steps [data-panel]').forEach(b=>{b.classList.toggle('active',b.dataset.panel===name);if(b.dataset.panel===name)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current')});
  if(name==='editor'){draw();window.onComparisonVisible?.()}
  else window.closePoreDetails?.();
  if(name==='load')setLoadMode('new');
  window.scrollTo(0,0);
}
window.updateWorkspaceControls=()=>{document.querySelector('[data-panel="setup"]').disabled=busy||(!setupProject&&!state);window.activeStopOperation?.render()};
document.querySelectorAll('[data-panel]').forEach(button=>button.onclick=()=>{
  if(button.dataset.panel==='setup'&&!setupProject&&state)return workspaceWork('Loading settings…',()=>prepareExisting(state.dataset));
  showPanel(button.dataset.panel);
});
function setLoadMode(mode){
  const fresh=mode==='new';
  $('newImagePane').classList.toggle('hidden',!fresh);$('savedImagePane').classList.toggle('hidden',fresh);
  for(const [id,active] of [['newImageTab',fresh],['openAnalysisTab',!fresh]]){$(id).classList.toggle('active',active);$(id).setAttribute('aria-selected',String(active))}
}
$('newImageTab').onclick=()=>setLoadMode('new');$('openAnalysisTab').onclick=()=>setLoadMode('saved');
function jobMessage(message,error=false){$('jobStatus').textContent=message;$('jobStatus').classList.toggle('error',error);$('imageSelectionStatus').textContent=message;$('imageSelectionStatus').classList.toggle('error',error)}
async function workspaceWork(message,fn){
  await work(message,async()=>{jobMessage(message);try{await fn()}catch(error){jobMessage(error.message,true);throw error}});
}
async function readImageLibrary(){
  const response=await fetch('/api/images');
  if(!response.ok)throw new Error('Could not load images. Retry. (HTTP '+response.status+')');
  let data;
  try{data=await response.json()}catch{throw new Error('Could not read image list. Retry.')}
  if(!Array.isArray(data?.images)||!data.images.every(entry=>entry&&typeof entry.id==='string'&&typeof entry.name==='string'&&typeof entry.preview_url==='string'&&Array.isArray(entry.analyses)&&entry.analyses.every(a=>a&&typeof a.dataset==='string')))
    throw new Error('Invalid image list. Refresh and retry.');
  return data.images;
}
function libraryFailed(error){$('libraryStatus').textContent=error.message;$('retryLibrary').classList.remove('hidden')}
function acceptLibrary(entries){imageLibrary=entries;$('libraryStatus').textContent='';$('retryLibrary').classList.add('hidden');updateHistory(state?.dataset??$('dataset').value);renderImageLibrary()}
async function refreshLists(){
  try{
    const [datasets,entries]=await Promise.all([fetch('/api/datasets').then(r=>r.json()),readImageLibrary()]);
    datasetLabels=datasets.labels??{};acceptLibrary(entries);
  }catch(error){libraryFailed(error);throw error}
}
$('retryLibrary').onclick=()=>workspaceWork('Loading images…',async()=>{await refreshLists();jobMessage('Images loaded.')});
function updateHistory(selected){
  const entry=imageLibrary.find(e=>e.analyses.some(a=>a.dataset===selected));
  $('dataset').replaceChildren();
  if(entry)entry.analyses.forEach(a=>$('dataset').add(new Option(entry.name+' · '+a.label+(a.revision?' · Revision '+a.revision:''),a.dataset)));
  else if(selected)$('dataset').add(new Option(datasetLabels[selected]??selected,selected));
  $('dataset').value=selected;
}
function renderImageLibrary(){
  $('imageLibrary').replaceChildren();
  for(const entry of imageLibrary.filter(entry=>entry.analyzed)){
    const card=document.createElement('button');card.className='image-card';card.dataset.imageId=entry.id;card.dataset.analyzed=String(entry.analyzed);
    const thumbnail=document.createElement('img');thumbnail.src=entry.preview_url;thumbnail.alt='';thumbnail.loading='lazy';
    const name=document.createElement('strong');name.textContent=entry.name;
    const detail=document.createElement('span');detail.className='hint';detail.textContent='Saved analysis';
    const action=document.createElement('span');action.className='image-action';action.textContent=entry.analyzed?'Open Analysis →':'Preprocess →';
    card.append(thumbnail,name,detail,action);card.onclick=()=>workspaceWork('Loading image…',async()=>selectImage(await api('open-image',{image_id:entry.id})));
    $('imageLibrary').append(card);
  }
  if(!$('imageLibrary').children.length){const message=document.createElement('p');message.className='hint';message.textContent='No saved analyses.';$('imageLibrary').append(message)}
  controls();
}
async function selectImage(selection){
  if(selection.dataset){
    previewGeneration++;previewPending=false;setupProject=null;setupImage=null;$('newAnalysisSettings').classList.add('hidden');
    await refreshLists();updateHistory(selection.dataset);
    await accept(await api('load',{dataset:selection.dataset}));showPanel('editor');
    jobMessage('Analysis loaded.');setStatus('Ready');
  }else{
    state=null;base=null;outlines=null;clear();
    await showProject(selection.project);
    jobMessage(selection.existing?'Check calibration.':'Check calibration.');
  }
}
function drawSetup(){
  setupContext.clearRect(0,0,setupCanvas.width,setupCanvas.height);
  if(!setupImage)return;
  setupContext.drawImage(setupImage,0,0);
  const bottom=Number($('analysisBottom').value);
  if(bottom>=0&&bottom<setupCanvas.height){
    setupContext.fillStyle='rgba(10,25,30,.35)';setupContext.fillRect(0,bottom,setupCanvas.width,setupCanvas.height-bottom);
    setupContext.strokeStyle='#46e7ee';setupContext.lineWidth=3;setupContext.beginPath();setupContext.moveTo(0,bottom);setupContext.lineTo(setupCanvas.width,bottom);setupContext.stroke();
  }
  setupContext.strokeStyle='#ffdc62';setupContext.fillStyle='#ffdc62';setupContext.lineWidth=3;
  scalePoints.forEach(([x,y])=>{setupContext.beginPath();setupContext.arc(x,y,5,0,Math.PI*2);setupContext.fill()});
  if(scalePoints.length===2){setupContext.beginPath();setupContext.moveTo(...scalePoints[0]);setupContext.lineTo(...scalePoints[1]);setupContext.stroke()}
}
function updateCalibration(){
  const um=Number($('scaleUm').value),px=Number($('scalePixels').value),minimum=Number($('minArea').value);
  const valid=um>0&&px>0;
  $('calibrationInfo').textContent=valid?'1px = '+(um/px).toPrecision(5)+' µm · Image area '+(setupProject.width*Number($('analysisBottom').value)*(um/px)**2).toLocaleString(undefined,{maximumFractionDigits:1})+' µm²':'Enter scale lengths.';
  $('minAreaInfo').textContent=valid?'Minimum: '+(minimum*(um/px)**2).toFixed(3)+' µm²':'';
}
async function showProject(project){
  $('advancedSettings').open=false;$('regionSettings').open=false;
  setupProject=project;scalePoints=[];measuringScale=false;$('measureScale').classList.remove('active');
  setupImage=await image(project.preview_url);setupCanvas.width=project.width;setupCanvas.height=project.height;
  $('setupImageTitle').textContent=project.name;
  $('imageDetails').textContent=project.width+' × '+project.height+'px · '+project.original_dtype+(project.normalization?' · 8-bit preview':'');
  $('analysisBottom').max=project.height;$('bottomSlider').max=project.height;
  const c=project.config;
  $('analysisBottom').value=$('bottomSlider').value=c.analysis_bottom;
  $('scaleUm').value=c.scale_um??'';$('scalePixels').value=c.scale_pixels??'';
  $('minContrast').value=c.min_contrast;$('minArea').value=c.min_area_pixels;$('pointDensity').value=c.points_per_side;
  const adjustable=c.preprocessing_mode==='adjustable';
  $('normalizeEnabled').checked=adjustable?c.normalize_enabled:['normalize','coarse'].includes(c.preprocessing_mode);$('backgroundStrength').value=adjustable?c.background_strength:({weak:4,medium:7,strong:10,detail:7}[c.coarse_strength]??0)*Number(['coarse','structure'].includes(c.preprocessing_mode));$('blurMethod').value=adjustable?c.blur_method:(['coarse','structure'].includes(c.preprocessing_mode)?'gaussian':'none');$('blurStrength').value=adjustable?c.blur_strength:3;invalidatePreprocessing();
  $('scaleConfirmed').checked=false;
  updateSettingsSummary();
  if(project.scale_detection){const s=project.scale_detection;scalePoints=[[s.x,s.y],[s.x+s.length_pixels,s.y]]}
  updateCalibration();drawSetup();await refreshLists();$('newAnalysisSettings').classList.remove('hidden');showPanel('setup');
  jobMessage('Confirm calibration to continue.');
}
$('uploadFile').onchange=()=>workspaceWork('Loading image…',async()=>{
  const file=$('uploadFile').files[0];if(!file)return;
  if(file.size>32*1024*1024)throw new Error('Choose an image up to 32 MB.');
  const content=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(new Error('Cannot read file.'));reader.readAsDataURL(file)});
  const selection=await api('upload',{name:file.name,content});
  if(selection.dataset){state=null;base=null;outlines=null;clear();await prepareExisting(selection.dataset)}else await selectImage(selection);$('uploadFile').value='';
});
async function prepareExisting(dataset){await showProject(await api('import-existing',{dataset}))}
for(const id of ['analysisBottom','bottomSlider'])$(id).oninput=()=>{if(!setupProject)return;const value=$(id).value;$('analysisBottom').value=$('bottomSlider').value=value;updateCalibration();drawSetup();invalidatePreprocessing()};
for(const id of ['scaleUm','scalePixels','minArea'])$(id).oninput=()=>{if(setupProject)updateCalibration();if(id!=='minArea')$('scaleConfirmed').checked=false};
$('resetCriteria').onclick=()=>{$('minContrast').value=8;$('minArea').value=100;$('pointDensity').value=48;$('normalizeEnabled').checked=true;$('backgroundStrength').value=0;$('blurMethod').value='none';$('blurStrength').value=2;invalidatePreprocessing();if(setupProject)updateCalibration()};
function preprocessingPayload(){return {analysis_bottom:Number($('analysisBottom').value),preprocessing_mode:'adjustable',normalize_enabled:$('normalizeEnabled').checked,background_strength:Number($('backgroundStrength').value),blur_method:$('blurMethod').value,blur_strength:Number($('blurStrength').value)}}
function updateSettingsSummary(){
  $('analysisSettingsSummary').textContent='Normalization '+($('normalizeEnabled').checked?'On':'Off')+' · Background '+$('backgroundStrength').value+' · '+$('blurMethod').selectedOptions[0].textContent;
}
for(const id of ['minContrast','minArea','pointDensity'])$(id).addEventListener('input',updateSettingsSummary);
let previewTimer=null,previewGeneration=0,previewInFlight=false,previewPending=false;
function invalidatePreprocessing(){
  previewGeneration++;previewPending=true;clearTimeout(previewTimer);
  $('backgroundValue').textContent=$('backgroundStrength').value==='0'?'Off':$('backgroundStrength').value;
  $('blurValue').textContent=$('blurStrength').value;$('blurStrengthField').classList.toggle('hidden',$('blurMethod').value==='none');updateSettingsSummary();
  if(!setupProject)return;
  $('preprocessingImage').style.opacity='.5';$('preprocessingInfo').textContent='Updating…';
  previewTimer=setTimeout(refreshPreprocessing,120);
}
async function refreshPreprocessing(){
  if(previewInFlight||!previewPending||!setupProject)return;
  previewInFlight=true;previewPending=false;
  const generation=previewGeneration,projectId=setupProject.id,config=preprocessingPayload();
  try{
    const data=await api('preprocess-preview',{project_id:projectId,config});
    const ready=await image(data.image);
    if(generation!==previewGeneration||setupProject?.id!==projectId)return;
    $('preprocessingImage').src=ready.src;$('preprocessingImage').style.opacity='1';
    $('preprocessingImage').dataset.generation=String(generation);
    $('preprocessingInfo').textContent='Updated · '+ready.width+' × '+ready.height+'px';
  }catch(error){if(generation===previewGeneration)$('preprocessingInfo').textContent=error.message;}
  finally{previewInFlight=false;if(previewPending)refreshPreprocessing();}
}
for(const id of ['normalizeEnabled','backgroundStrength','blurMethod','blurStrength'])$(id).addEventListener('input',invalidatePreprocessing);
$('measureScale').onclick=()=>{if(!setupProject){jobMessage('Choose an image first.',true);return}measuringScale=!measuringScale;scalePoints=[];$('measureScale').classList.toggle('active',measuringScale);drawSetup();jobMessage('Click both ends of the scale bar.')};
$('scaleConfirmed').onchange=()=>{if($('scaleConfirmed').checked)jobMessage('Ready to analyze.')};
setupCanvas.onclick=event=>{
  if(busy||!setupProject||!measuringScale)return;
  const r=setupCanvas.getBoundingClientRect(),x=Math.max(0,Math.min(setupCanvas.width-1,(event.clientX-r.left)*setupCanvas.width/r.width)),y=Math.max(0,Math.min(setupCanvas.height-1,(event.clientY-r.top)*setupCanvas.height/r.height));
  scalePoints.push([x,y]);
  if(scalePoints.length===2){$('scalePixels').value=Math.hypot(scalePoints[1][0]-scalePoints[0][0],scalePoints[1][1]-scalePoints[0][1]).toFixed(2);measuringScale=false;$('measureScale').classList.remove('active');$('scaleConfirmed').checked=false;updateCalibration();jobMessage('Enter the scale length in µm and confirm.')}
  drawSetup();
};
function analysisDuration(seconds){
  seconds=Math.max(0,Math.ceil(seconds));
  if(seconds<60)return seconds+'s';
  if(seconds<3600)return Math.floor(seconds/60)+'m '+(seconds%60)+'s';
  return Math.floor(seconds/3600)+'h '+Math.floor(seconds%3600/60)+'m';
}
function analysisJobMessage(job){
  if(!Number.isFinite(job.elapsed_seconds))return job.message;
  let message=job.message+' · Elapsed '+analysisDuration(job.elapsed_seconds);
  if(job.status==='running'&&job.progress<82){
    message+=Number.isFinite(job.detection_remaining_seconds)
      ?' · Detection: ~'+analysisDuration(job.detection_remaining_seconds)+' remaining'
      :' · Estimating time…';
  }
  return message;
}
async function followJob(jobId){
  localStorage.setItem('poreActiveJob',jobId);
  const operation=window.activeStopOperation||beginStopOperation('runAnalysis','Run Analysis',jobMessage);
  operation.connect(()=>api('stop-analysis',{job_id:jobId}));
  try{
  while(true){
    const job=await api('job',{job_id:jobId});$('analysisProgress').value=job.progress;jobMessage(analysisJobMessage(job));
    if(job.status==='stopping')operation.markStopping();
    if(job.status==='cancelled'){localStorage.removeItem('poreActiveJob');setStatus('Analysis stopped.');return}
    if(job.status==='failed'){localStorage.removeItem('poreActiveJob');throw new Error(job.message)}
    if(job.status==='complete'){
      localStorage.removeItem('poreActiveJob');await refreshLists();updateHistory(job.dataset);
      await accept(await api('load',{dataset:job.dataset}));showPanel('editor');
      jobMessage('Analysis complete · '+job.candidate_count+' pores'+(Number.isFinite(job.elapsed_seconds)?' · Elapsed '+analysisDuration(job.elapsed_seconds):''));setStatus('Analysis complete.');return;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  }finally{operation.finish()}
}
function validateAnalysisInput(){
  if(!setupProject)throw new Error('Choose an image first.');
  const fields={scaleUm:'Enter the scale length in µm.',scalePixels:'Measure or enter the scale length in pixels.',analysisBottom:'Check the bottom boundary.',minContrast:'Enter contrast from 0 to 255.',minArea:'Enter a minimum pore area.'};
  for(const [id,message] of Object.entries(fields)){
    const field=$(id);
    if(field.value===''||!field.checkValidity()){
      const section=field.closest('details');if(section)section.open=true;
      field.focus();throw new Error(message);
    }
  }
  if(!$('scaleConfirmed').checked){$('scaleConfirmed').focus();throw new Error('Confirm calibration first.')}
}
$('runAnalysis').onclick=()=>{
  if(window.activeStopOperation?.buttonId==='runAnalysis')return window.activeStopOperation.stop();
  if(busy)return;
  try{validateAnalysisInput()}catch(error){jobMessage(error.message,true);setStatus(error.message,true);return}
  return workspaceWork('Preparing analysis…',async()=>{
  const operation=beginStopOperation('runAnalysis','Run Analysis',jobMessage);
  try{
  const config={...preprocessingPayload(),scale_um:Number($('scaleUm').value),scale_pixels:Number($('scalePixels').value),min_contrast:Number($('minContrast').value),min_area_pixels:Number($('minArea').value),points_per_side:Number($('pointDensity').value),scale_confirmed:$('scaleConfirmed').checked};
  const job=await api('analyze',{project_id:setupProject.id,config});$('analysisProgress').value=0;await followJob(job.job_id);
  }finally{operation.finish()}
  });
};
window.onEditorAccepted=data=>{
  updateHistory(data.dataset);
  const imageName=imageLibrary.find(e=>e.analyses.some(a=>a.dataset===data.dataset))?.name??datasetLabels[data.dataset]??data.dataset;
  $('editorImageName').textContent=imageName;
  $('editorImageName').title=imageName;
  $('analysisTitle').textContent='Generate Report · '+imageName;
  $('analysisSubtitle').textContent=data.stats.candidate_count+' pores · Area fraction (2D) '+data.stats.candidate_union_area_percent.toFixed(2)+'%';
  const ready=Boolean(data.report_ready??data.report_url);
  $('analysisFrame').classList.toggle('hidden',!ready);$('analysisComparison').closest('section').classList.toggle('hidden',!ready);
  if(ready){$('analysisFrame').src=data.report_url;$('analysisComparison').src=data.image_url}
  else{$('analysisFrame').removeAttribute('src');$('analysisComparison').removeAttribute('src')}
  if(!$('exportDirectory').value){try{$('exportDirectory').value=localStorage.getItem('poreExportDirectory')||data.export_default_directory||''}catch{$('exportDirectory').value=data.export_default_directory||''}}
  $('generateReport').textContent=ready?'Save Report':'Generate Report';
  if($('exportResult').dataset.revisionKey!==data.dataset+':'+data.revision)$('exportResult').textContent='';
  $('reportGenerationStatus').textContent=ready?'Report ready.':'Ready to generate.';
  $('analysisDownloads').replaceChildren();
  for(const [name,file] of [['Pore CSV','candidates.csv'],['Diameter CSV','diameter_histogram.csv'],['Area CSV','area_histogram.csv'],['Spatial CSV','spatial_grid.csv'],['Summary JSON','summary.json'],['PDF Report','dashboard.pdf']]){
    if(ready){const a=document.createElement('a');a.textContent=name;a.href=data.result_base+'/measurements/'+file;a.download=file;$('analysisDownloads').append(a)}
  }
  // Refresh badges after manual saves without reloading the active mask.
  readImageLibrary().then(acceptLibrary).catch(libraryFailed);
};
$('editFromAnalysis').onclick=()=>showPanel('editor');
$('generateReport').onclick=()=>work('Generating report…',async()=>{
  const directory=$('exportDirectory').value.trim();
  if(!directory){$('exportDirectory').focus();throw new Error('Choose an export folder.')}
  $('reportGenerationStatus').textContent='Generating report and images…';
  try{const data=await api('generate-report',{...payload(),export_directory:directory});await accept(data);$('exportResult').textContent='Exported to: '+data.exported_folder;$('exportResult').dataset.revisionKey=data.dataset+':'+data.revision;try{localStorage.setItem('poreExportDirectory',directory)}catch{}setStatus('Report saved.')}
  catch(error){$('reportGenerationStatus').textContent='Export failed: '+error.message+' Please retry.';throw error}
});
$('chooseExportDirectory').onclick=()=>work('Choose a folder.',async()=>{const result=await api('choose-export-folder',{});if(result.directory)$('exportDirectory').value=result.directory;setStatus('Export folder selected.')});
window.updateToolLayout=()=>{
  const selecting=mode==='select',cutting=mode==='cut';
  $('selectionActions').classList.toggle('hidden',!selecting);
  $('cutPanel').classList.toggle('hidden',!cutting);
  $('regionPanel').classList.toggle('hidden',selecting||cutting);
  $('previewControls').classList.toggle('hidden',selecting||cutting);
  document.querySelector('.tool-history').classList.toggle('hidden',selecting);
};
window.updateToolLayout();
$('editFromAnalysis').textContent='Back to Editor';
$('settingsFromAnalysis').textContent='Reprocess Image';
$('settingsFromAnalysis').onclick=()=>workspaceWork('Loading settings…',()=>prepareExisting(state.dataset));
(async()=>{try{while(busy)await new Promise(resolve=>setTimeout(resolve,50));await refreshLists();if(state)window.onEditorAccepted(state);const job=localStorage.getItem('poreActiveJob');if(job)await workspaceWork('Checking active analysis…',()=>followJob(job))}catch(error){jobMessage(error.message,true)}})();

$('saveEditorImages').onclick=()=>work('Choose an image destination folder.',async()=>{
  if(!state)return;
  const picked=await api('choose-export-folder',{});
  if(!picked.directory){setStatus('Image export canceled.');return;}
  setStatus('Saving images...');
  const result=await api('export-images',{...payload(),directory:picked.directory,color:$('poreColor').value,opacity:Number($('poreOpacity').value),labels:$('showPoreLabels').checked});
  setStatus('3 images saved: '+result.exported_folder);
});
