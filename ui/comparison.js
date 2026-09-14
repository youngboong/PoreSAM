// Display preferences change only canvas rendering, never saved masks or measurements.
(() => {
  const original=$('originalImage'),originalContext=original.getContext('2d');
  const left=$('originalViewport'),right=$('viewport');
  let supplemental=null,fill=null,edges=null,numbers=null,lastBase=null,dataset=null,autoFit=true;
  const tintedSupplemental=document.createElement('canvas');
  window.supplementalColor=()=>{
    const base=$('poreColor').value;
    const distance=color=>[1,3,5].reduce((n,i)=>n+(parseInt(base.slice(i,i+2),16)-parseInt(color.slice(i,i+2),16))**2,0);
    return ['#00cde0','#9654e8','#27c66f'].sort((a,b)=>distance(b)-distance(a))[0];
  };
  const tintedFill=document.createElement('canvas'),tintedEdges=document.createElement('canvas');
  try{
    const settings=JSON.parse(localStorage.getItem('poreDisplay')||'{}');
    if(/^#[0-9a-f]{6}$/i.test(settings.color))$('poreColor').value=settings.color;
    if(Number.isFinite(settings.opacity)&&settings.opacity>=0&&settings.opacity<=100)$('poreOpacity').value=settings.opacity;
  }catch{}
  function tint(source,target,color=$('poreColor').value){
    target.width=canvas.width;target.height=canvas.height;
    if(!source)return;
    const c=target.getContext('2d');c.drawImage(source,0,0);
    c.globalCompositeOperation='source-in';c.fillStyle=color;c.fillRect(0,0,target.width,target.height);
    c.globalCompositeOperation='source-over';
  }
  function updateColor(){tint(supplemental,tintedSupplemental,window.supplementalColor());$('supplementalLegendDot').style.background=window.supplementalColor();tint(fill,tintedFill);tint(edges,tintedEdges);$('poreLegendDot').style.background=$('poreColor').value;}
  function savePreferences(){
    $('poreOpacityValue').textContent=$('poreOpacity').value+'%';
    try{localStorage.setItem('poreDisplay',JSON.stringify({color:$('poreColor').value,opacity:Number($('poreOpacity').value)}))}catch{}
  }
  window.loadComparisonLayers=async data=>{
    [supplemental,fill,edges,numbers]=await Promise.all([data.supplemental_overlay?image(data.supplemental_overlay):null,data.fill_overlay?image(data.fill_overlay):null,data.edge_overlay?image(data.edge_overlay):null,data.label_overlay?image(data.label_overlay):null]);
    if(dataset!==data.dataset){autoFit=true;left.scrollTo(0,0);right.scrollTo(0,0)}
    dataset=data.dataset;original.width=data.width;original.height=data.height;lastBase=null;
    updateColor();
  };
  window.drawOriginalComparison=()=>{
    if(lastBase===base)return;
    originalContext.clearRect(0,0,original.width,original.height);
    if(base)originalContext.drawImage(base,0,0);
    lastBase=base;
  };
  window.drawPoreLayer=context=>{
    if(!$('showOverlay').checked)return;
    if(!fill){if(outlines)context.drawImage(outlines,0,0);return;}
    context.save();context.globalAlpha=Number($('poreOpacity').value)/100;context.drawImage(tintedFill,0,0);context.drawImage(tintedSupplemental,0,0);
    context.globalAlpha=1;context.drawImage(tintedEdges,0,0);
    if(numbers&&$('showPoreLabels').checked)context.drawImage(numbers,0,0);
    context.restore();
  };
  function applyZoom(){
    const zoom=Number($('zoom').value)/100;
    for(const c of [canvas,original]){c.style.width=(canvas.width*zoom)+'px';c.style.height=(canvas.height*zoom)+'px';}
    $('zoomLabel').textContent=$('zoom').value+'%';
  }
  function fit(){
    if(!base||!right.clientWidth||!right.clientHeight)return;
    const ratio=Math.min((Math.min(left.clientWidth,right.clientWidth)-18)/canvas.width,(Math.min(left.clientHeight,right.clientHeight)-18)/canvas.height,1);
    $('zoom').value=Math.max(10,Math.floor(ratio*100));applyZoom();
  }
  window.resizeComparison=()=>{if(autoFit)fit();else applyZoom()};
  window.onComparisonVisible=()=>requestAnimationFrame(window.resizeComparison);
  $('zoom').oninput=()=>{autoFit=false;applyZoom()};
  $('fitComparison').onclick=()=>{autoFit=true;fit()};
  $('poreColor').oninput=()=>{updateColor();savePreferences();draw()};
  $('poreOpacity').oninput=()=>{savePreferences();draw()};
  $('showPoreLabels').onchange=draw;
  for(const [source,target] of [[left,right],[right,left]])source.addEventListener('scroll',()=>{
    if(target.scrollLeft!==source.scrollLeft)target.scrollLeft=source.scrollLeft;
    if(target.scrollTop!==source.scrollTop)target.scrollTop=source.scrollTop;
  },{passive:true});
  window.addEventListener('resize',()=>{if(autoFit)requestAnimationFrame(fit)});
  $('poreOpacityValue').textContent=$('poreOpacity').value+'%';
  $('poreLegendDot').style.background=$('poreColor').value;
})();
