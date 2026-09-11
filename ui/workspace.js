let setupProject=null,setupImage=null,scalePoints=[],measuringScale=false,imageLibrary=[],datasetLabels={};
const setupCanvas=$('setupCanvas'),setupContext=setupCanvas.getContext('2d');
function showPanel(name){
  if(!['load','setup','editor','analysis'].includes(name))return;
  if(['editor','analysis'].includes(name)&&!state)return;
  if(name==='setup'&&!setupProject)return;
  for(const panel of ['load','setup','editor','analysis'])$(panel+'Panel').classList.toggle('hidden',panel!==name);
  document.querySelectorAll('.steps [data-panel]').forEach(b=>{b.classList.toggle('active',b.dataset.panel===name);if(b.dataset.panel===name)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current')});
  if(name==='editor'){draw();window.onComparisonVisible?.()}
  else window.closePoreDetails?.();
  window.scrollTo(0,0);
}
window.updateWorkspaceControls=()=>{document.querySelector('[data-panel="setup"]').disabled=busy||(!setupProject&&!state)};
document.querySelectorAll('[data-panel]').forEach(button=>button.onclick=()=>{
  if(button.dataset.panel==='setup'&&!setupProject&&state)return workspaceWork('현재 이미지의 전처리 설정을 불러옵니다.',()=>prepareExisting(state.dataset));
  showPanel(button.dataset.panel);
});
function jobMessage(message,error=false){$('jobStatus').textContent=message;$('jobStatus').classList.toggle('error',error);$('imageSelectionStatus').textContent=message;$('imageSelectionStatus').classList.toggle('error',error)}
async function workspaceWork(message,fn){
  await work(message,async()=>{jobMessage(message);try{await fn()}catch(error){jobMessage(error.message,true);throw error}});
}
async function readImageLibrary(){
  const response=await fetch('/api/images');
  if(!response.ok)throw new Error('이미지 목록을 불러오지 못했습니다. 다시 불러오기를 눌러주세요. (HTTP '+response.status+')');
  let data;
  try{data=await response.json()}catch{throw new Error('이미지 목록 응답을 읽지 못했습니다. 다시 불러오기를 눌러주세요.')}
  if(!Array.isArray(data?.images)||!data.images.every(entry=>entry&&typeof entry.id==='string'&&typeof entry.name==='string'&&typeof entry.preview_url==='string'&&Array.isArray(entry.analyses)&&entry.analyses.every(a=>a&&typeof a.dataset==='string')))
    throw new Error('이미지 목록 응답이 올바르지 않습니다. 화면을 새로고침하거나 목록을 다시 불러와주세요.');
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
$('retryLibrary').onclick=()=>workspaceWork('이미지 목록을 다시 불러오고 있습니다.',async()=>{await refreshLists();jobMessage('이미지 목록을 불러왔습니다.')});
function updateHistory(selected){
  const entry=imageLibrary.find(e=>e.analyses.some(a=>a.dataset===selected));
  $('dataset').replaceChildren();
  if(entry)entry.analyses.forEach(a=>$('dataset').add(new Option(entry.name+' · '+a.label+(a.revision?' · 수정본 '+a.revision:''),a.dataset)));
  else if(selected)$('dataset').add(new Option(datasetLabels[selected]??selected,selected));
  $('dataset').value=selected;
}
function renderImageLibrary(){
  $('imageLibrary').replaceChildren();$('pendingImages').replaceChildren();
  for(const entry of imageLibrary){
    const card=document.createElement('button');card.className='image-card';card.dataset.imageId=entry.id;card.dataset.analyzed=String(entry.analyzed);
    const thumbnail=document.createElement('img');thumbnail.src=entry.preview_url;thumbnail.alt='';thumbnail.loading='lazy';
    const name=document.createElement('strong');name.textContent=entry.name;
    const detail=document.createElement('span');detail.className='hint';detail.textContent=entry.analyzed?'분석 '+entry.analysis_count+'개'+(entry.revision?' · 수정본 '+entry.revision:'')+' · 결과 분석에서 열기':'분석 영역과 스케일 설정하기';
    const action=document.createElement('span');action.className='image-action';action.textContent=entry.analyzed?'분석된 이미지 불러오기 →':'전처리 이어가기 →';
    card.append(thumbnail,name,detail,action);card.onclick=()=>workspaceWork('이미지를 불러오고 있습니다.',async()=>selectImage(await api('open-image',{image_id:entry.id})));
    $(entry.analyzed?'imageLibrary':'pendingImages').append(card);
  }
  $('pendingImagesSection').classList.toggle('hidden',!$('pendingImages').children.length);
  if(!$('imageLibrary').children.length){const message=document.createElement('p');message.className='hint';message.textContent='분석한 이미지가 여기에 표시됩니다.';$('imageLibrary').append(message)}
  controls();
}
async function selectImage(selection){
  if(selection.dataset){
    previewGeneration++;previewPending=false;setupProject=null;setupImage=null;$('newAnalysisSettings').classList.add('hidden');
    await refreshLists();updateHistory(selection.dataset);
    await accept(await api('load',{dataset:selection.dataset}));showPanel('editor');
    jobMessage('저장된 분석과 최신 수정본을 불러왔습니다.');setStatus('저장된 분석을 이어서 수정할 수 있습니다.');
  }else{
    state=null;base=null;outlines=null;clear();
    await showProject(selection.project);
    jobMessage(selection.existing?'아직 분석하지 않은 이미지입니다. 설정을 확인해주세요.':'새 이미지입니다. 분석 영역과 스케일을 설정해주세요.');
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
  $('calibrationInfo').textContent=valid?'1px = '+(um/px).toPrecision(5)+' µm · 분석 면적 '+(setupProject.width*Number($('analysisBottom').value)*(um/px)**2).toLocaleString(undefined,{maximumFractionDigits:1})+' µm²':'실제 길이와 픽셀 길이를 입력하세요.';
  $('minAreaInfo').textContent=valid?'현재 스케일에서 '+(minimum*(um/px)**2).toFixed(3)+' µm² 이상인 후보를 선택합니다.':'기본값 100px. 스케일에 따라 실제 최소 면적이 달라집니다.';
}
async function showProject(project){
  $('advancedSettings').open=false;$('regionSettings').open=false;
  setupProject=project;scalePoints=[];measuringScale=false;$('measureScale').classList.remove('active');
  setupImage=await image(project.preview_url);setupCanvas.width=project.width;setupCanvas.height=project.height;
  $('setupImageTitle').textContent=project.name;
  $('imageDetails').textContent=project.width+' × '+project.height+'px · '+project.original_dtype+(project.normalization?' · 밝기 범위를 0~255로 변환해 분석합니다.':'');
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
  jobMessage('분석 영역과 스케일바 실제 길이를 확인한 뒤 실행하세요.');
}
$('uploadFile').onchange=()=>workspaceWork('이미지를 불러오고 있습니다.',async()=>{
  const file=$('uploadFile').files[0];if(!file)return;
  if(file.size>32*1024*1024)throw new Error('32MB 이하의 이미지 파일을 선택해주세요.');
  const content=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(new Error('파일을 읽을 수 없습니다.'));reader.readAsDataURL(file)});
  await selectImage(await api('upload',{name:file.name,content}));$('uploadFile').value='';
});
async function prepareExisting(dataset){await showProject(await api('import-existing',{dataset}))}
$('reuseImage').onclick=()=>workspaceWork('기존 이미지로 새 분석을 준비하고 있습니다.',()=>prepareExisting(state.dataset));
for(const id of ['analysisBottom','bottomSlider'])$(id).oninput=()=>{if(!setupProject)return;const value=$(id).value;$('analysisBottom').value=$('bottomSlider').value=value;updateCalibration();drawSetup();invalidatePreprocessing()};
for(const id of ['scaleUm','scalePixels','minArea'])$(id).oninput=()=>{if(setupProject)updateCalibration();if(id!=='minArea')$('scaleConfirmed').checked=false};
$('resetCriteria').onclick=()=>{$('minContrast').value=8;$('minArea').value=100;$('pointDensity').value=48;$('normalizeEnabled').checked=true;$('backgroundStrength').value=0;$('blurMethod').value='none';$('blurStrength').value=2;invalidatePreprocessing();if(setupProject)updateCalibration()};
function preprocessingPayload(){return {analysis_bottom:Number($('analysisBottom').value),preprocessing_mode:'adjustable',normalize_enabled:$('normalizeEnabled').checked,background_strength:Number($('backgroundStrength').value),blur_method:$('blurMethod').value,blur_strength:Number($('blurStrength').value)}}
function updateSettingsSummary(){
  $('analysisSettingsSummary').textContent='밝기 정규화 '+($('normalizeEnabled').checked?'켜짐':'꺼짐')+' · Background '+$('backgroundStrength').value+' · '+$('blurMethod').selectedOptions[0].textContent;
}
for(const id of ['minContrast','minArea','pointDensity'])$(id).addEventListener('input',updateSettingsSummary);
let previewTimer=null,previewGeneration=0,previewInFlight=false,previewPending=false;
function invalidatePreprocessing(){
  previewGeneration++;previewPending=true;clearTimeout(previewTimer);
  $('backgroundValue').textContent=$('backgroundStrength').value==='0'?'0 · 끄기':$('backgroundStrength').value;
  $('blurValue').textContent=$('blurStrength').value;$('blurStrengthField').classList.toggle('hidden',$('blurMethod').value==='none');updateSettingsSummary();
  if(!setupProject)return;
  $('preprocessingImage').style.opacity='.5';$('preprocessingInfo').textContent='변경한 설정을 적용하는 중…';
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
    $('preprocessingInfo').textContent='미리보기 반영 완료 · '+ready.width+' × '+ready.height+'px';
  }catch(error){if(generation===previewGeneration)$('preprocessingInfo').textContent=error.message;}
  finally{previewInFlight=false;if(previewPending)refreshPreprocessing();}
}
for(const id of ['normalizeEnabled','backgroundStrength','blurMethod','blurStrength'])$(id).addEventListener('input',invalidatePreprocessing);
$('measureScale').onclick=()=>{if(!setupProject){jobMessage('먼저 이미지를 선택해주세요.',true);return}measuringScale=!measuringScale;scalePoints=[];$('measureScale').classList.toggle('active',measuringScale);drawSetup();jobMessage('스케일바의 양 끝을 차례로 클릭하세요.')};
$('clearScaleLine').onclick=()=>{scalePoints=[];measuringScale=false;$('measureScale').classList.remove('active');drawSetup()};
$('scaleConfirmed').onchange=()=>{if($('scaleConfirmed').checked)jobMessage('설정을 확인한 뒤 자동 분석을 실행하세요.')};
setupCanvas.onclick=event=>{
  if(busy||!setupProject||!measuringScale)return;
  const r=setupCanvas.getBoundingClientRect(),x=Math.max(0,Math.min(setupCanvas.width-1,(event.clientX-r.left)*setupCanvas.width/r.width)),y=Math.max(0,Math.min(setupCanvas.height-1,(event.clientY-r.top)*setupCanvas.height/r.height));
  scalePoints.push([x,y]);
  if(scalePoints.length===2){$('scalePixels').value=Math.hypot(scalePoints[1][0]-scalePoints[0][0],scalePoints[1][1]-scalePoints[0][1]).toFixed(2);measuringScale=false;$('measureScale').classList.remove('active');$('scaleConfirmed').checked=false;updateCalibration();jobMessage('픽셀 길이를 측정했습니다. 스케일바에 적힌 실제 길이를 입력하고 확인해주세요.')}
  drawSetup();
};
async function followJob(jobId){
  localStorage.setItem('poreActiveJob',jobId);
  while(true){
    const job=await api('job',{job_id:jobId});$('analysisProgress').value=job.progress;jobMessage(job.message);
    if(job.status==='failed'){localStorage.removeItem('poreActiveJob');throw new Error(job.message)}
    if(job.status==='complete'){
      localStorage.removeItem('poreActiveJob');await refreshLists();updateHistory(job.dataset);
      await accept(await api('load',{dataset:job.dataset}));showPanel('editor');
      jobMessage('분석 완료 · '+job.candidate_count+'개 후보. 결과 분석에서 누락과 경계를 보완하세요.');setStatus('분석을 완료했습니다. pore를 확인·수정한 뒤 결과 보고서를 열어주세요.');return;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
}
function validateAnalysisInput(){
  if(!setupProject)throw new Error('먼저 이미지를 선택해주세요.');
  const fields={scaleUm:'스케일바에 적힌 실제 길이를 입력해주세요.',scalePixels:'스케일바의 양 끝을 찍거나 픽셀 길이를 입력해주세요.',analysisBottom:'분석 영역의 하단 위치를 확인해주세요.',minContrast:'밝기 차이를 0~255 사이로 입력해주세요.',minArea:'최소 후보 면적을 입력해주세요.'};
  for(const [id,message] of Object.entries(fields)){
    const field=$(id);
    if(field.value===''||!field.checkValidity()){
      const section=field.closest('details');if(section)section.open=true;
      field.focus();throw new Error(message);
    }
  }
  if(!$('scaleConfirmed').checked){$('scaleConfirmed').focus();throw new Error('이미지의 스케일바 길이가 맞으면 확인란을 체크해주세요.')}
}
$('runAnalysis').onclick=()=>{
  if(busy)return;
  try{validateAnalysisInput()}catch(error){jobMessage(error.message,true);setStatus(error.message,true);return}
  return workspaceWork('분석을 준비하고 있습니다.',async()=>{
  const config={...preprocessingPayload(),scale_um:Number($('scaleUm').value),scale_pixels:Number($('scalePixels').value),min_contrast:Number($('minContrast').value),min_area_pixels:Number($('minArea').value),points_per_side:Number($('pointDensity').value),scale_confirmed:$('scaleConfirmed').checked};
  const job=await api('analyze',{project_id:setupProject.id,config});$('analysisProgress').value=0;await followJob(job.job_id);
  });
};
window.onEditorAccepted=data=>{
  updateHistory(data.dataset);
  $('analysisTitle').textContent='결과 보고서 · '+($('dataset').selectedOptions[0]?.textContent??data.dataset)+' · '+(data.revision?'수정본 '+data.revision:'자동 분석');
  $('analysisSubtitle').textContent='후보 '+data.stats.candidate_count+'개 · 면적률 '+data.stats.candidate_union_area_percent.toFixed(2)+'% · 현재 결과의 면적·직경·공간 분포';
  $('analysisFrame').src=data.report_url;$('analysisComparison').src=data.image_url;
  $('analysisDownloads').replaceChildren();
  for(const [name,file] of [['후보별 CSV','candidates.csv'],['직경 분포 CSV','diameter_histogram.csv'],['면적 분포 CSV','area_histogram.csv'],['영역별 CSV','spatial_grid.csv'],['요약 JSON','summary.json'],['PDF 보고서','dashboard.pdf']]){
    const a=document.createElement('a');a.textContent=name;a.href=data.result_base+'/measurements/'+file;a.download=file;$('analysisDownloads').append(a);
  }
  $('report').textContent='결과 보고서 보기 →';$('report').onclick=e=>{e.preventDefault();showPanel('analysis')};
  // Refresh badges after manual saves without reloading the active mask.
  readImageLibrary().then(acceptLibrary).catch(libraryFailed);
};
$('editFromAnalysis').onclick=()=>showPanel('editor');
$('editFromAnalysis').textContent='결과 분석으로 돌아가기';
$('settingsFromAnalysis').textContent='전처리 다시 설정';
$('settingsFromAnalysis').onclick=()=>workspaceWork('이 이미지로 새 분석을 준비하고 있습니다.',()=>prepareExisting(state.dataset));
(async()=>{try{while(busy)await new Promise(resolve=>setTimeout(resolve,50));await refreshLists();if(state)window.onEditorAccepted(state);const job=localStorage.getItem('poreActiveJob');if(job)await workspaceWork('진행 중인 분석을 확인하고 있습니다.',()=>followJob(job))}catch(error){jobMessage(error.message,true)}})();
