// Display preferences change only canvas rendering, never saved masks or measurements.
(() => {
  const original=$('originalImage'),originalContext=original.getContext('2d');
  const left=$('originalViewport'),right=$('viewport');
  let supplemental=null,fill=null,edges=null,numbers=null,lastBase=null,dataset=null,autoFit=true;
  const panes=[{view:left,image:original},{view:right,image:canvas}];
  let cursor=null,cursorFrame=0;
  for(const pane of panes){
    pane.pointer=document.createElement('div');
    pane.pointer.className='comparison-cursor-overlay hidden';
    pane.pointer.setAttribute('aria-hidden','true');
    pane.pointer.innerHTML='<svg class="comparison-cursor" width="30" height="30" viewBox="0 0 30 30"><path d="M15 1V9 M15 21V29 M1 15H9 M21 15H29" fill="none" stroke="#111" stroke-width="5"/><path d="M15 1V9 M15 21V29 M1 15H9 M21 15H29" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="15" cy="15" r="4" fill="currentColor" stroke="#111" stroke-width="2"/></svg>';
    pane.view.parentElement.append(pane.pointer);
  }
  function pointerColor(){
    const occupied=[$('poreColor').value,window.supplementalColor(),window.selectionColor?.()??'#ff00ff'];
    const distance=color=>Math.min(...occupied.map(base=>[1,3,5].reduce((sum,i)=>sum+(parseInt(color.slice(i,i+2),16)-parseInt(base.slice(i,i+2),16))**2,0)));
    return ['#ff7a00','#00ff66','#00a6ff','#ff285f','#e6ff00','#b34dff','#00ffd5'].sort((a,b)=>distance(b)-distance(a))[0];
  }
  function refreshCursor(){
    cursorFrame=0;
    for(const pane of panes)pane.pointer.classList.add('hidden');
    if(!cursor||!base||!state)return;
    const source=cursor.pane,rect=source.image.getBoundingClientRect(),viewRect=source.view.getBoundingClientRect();
    if(!rect.width||!rect.height||cursor.x<viewRect.left+source.view.clientLeft||cursor.x>=viewRect.left+source.view.clientLeft+source.view.clientWidth||cursor.y<viewRect.top+source.view.clientTop||cursor.y>=viewRect.top+source.view.clientTop+source.view.clientHeight)return;
    const x=(cursor.x-rect.left)/rect.width,y=(cursor.y-rect.top)/rect.height;
    if(x<0||x>=1||y<0||y>=1)return;
    const target=panes.find(p=>p!==source),targetRect=target.image.getBoundingClientRect(),targetView=target.view.getBoundingClientRect();
    const tx=targetRect.left+x*targetRect.width-targetView.left-target.view.clientLeft;
    const ty=targetRect.top+y*targetRect.height-targetView.top-target.view.clientTop;
    if(!target.view.clientWidth||tx<0||tx>=target.view.clientWidth||ty<0||ty>=target.view.clientHeight)return;
    Object.assign(target.pointer.style,{left:(target.view.offsetLeft+target.view.clientLeft)+'px',top:(target.view.offsetTop+target.view.clientTop)+'px',width:target.view.clientWidth+'px',height:target.view.clientHeight+'px',color:pointerColor()});
    target.pointer.firstElementChild.style.transform=`translate(${tx-15}px,${ty-15}px)`;
    target.pointer.classList.remove('hidden');
  }
  function updateCursor(){if(!cursorFrame)cursorFrame=requestAnimationFrame(refreshCursor)}
  function clearCursor(){cursor=null;updateCursor()}
  for(const pane of panes){
    pane.image.addEventListener('pointermove',event=>{cursor={pane,x:event.clientX,y:event.clientY};updateCursor()});
    pane.image.addEventListener('pointerleave',clearCursor);
    pane.image.addEventListener('pointercancel',clearCursor);
    pane.view.addEventListener('wheel',event=>{
      if(!base||!state||!event.deltaY)return;
      event.preventDefault();
      // Keep an in-progress drawing anchored until the mouse button is released.
      if(busy||drag||!event.deltaY)return;
      const control=$('zoom'),before=Number(control.value),rect=pane.image.getBoundingClientRect(),viewRect=pane.view.getBoundingClientRect();
      const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?pane.view.clientHeight:1);
      let next=Math.round(before*Math.exp(-Math.max(-240,Math.min(240,delta))*.002));
      if(next===before)next+=delta<0?1:-1;
      next=Math.max(Number(control.min),Math.min(Number(control.max),next));
      if(next===before)return;
      const x=(event.clientX-rect.left)*pane.image.width/rect.width,y=(event.clientY-rect.top)*pane.image.height/rect.height;
      const offsetX=event.clientX-viewRect.left-pane.view.clientLeft,offsetY=event.clientY-viewRect.top-pane.view.clientTop;
      autoFit=false;control.value=next;applyZoom();
      pane.view.scrollLeft=x*next/100-offsetX;pane.view.scrollTop=y*next/100-offsetY;
      for(const other of panes)if(other!==pane)other.view.scrollTo(pane.view.scrollLeft,pane.view.scrollTop);
      cursor={pane,x:event.clientX,y:event.clientY};updateCursor();
    },{passive:false});
  }
  window.addEventListener('blur',clearCursor);
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
  function updateColor(){tint(supplemental,tintedSupplemental,window.supplementalColor());$('supplementalLegendDot').style.background=window.supplementalColor();tint(fill,tintedFill);tint(edges,tintedEdges);$('poreLegendDot').style.background=$('poreColor').value;updateCursor();}
  function savePreferences(){
    $('poreOpacityValue').textContent=$('poreOpacity').value+'%';
    try{localStorage.setItem('poreDisplay',JSON.stringify({color:$('poreColor').value,opacity:Number($('poreOpacity').value)}))}catch{}
  }
  window.loadComparisonLayers=async data=>{
    [supplemental,fill,edges,numbers]=await Promise.all([data.supplemental_overlay?image(data.supplemental_overlay):null,data.fill_overlay?image(data.fill_overlay):null,data.edge_overlay?image(data.edge_overlay):null,data.label_overlay?image(data.label_overlay):null]);
    if(dataset!==data.dataset){autoFit=true;clearCursor();left.scrollTo(0,0);right.scrollTo(0,0)}
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
    updateCursor();
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
    updateCursor();
  },{passive:true});
  window.addEventListener('resize',()=>{if(autoFit)requestAnimationFrame(fit);updateCursor()});
  $('poreOpacityValue').textContent=$('poreOpacity').value+'%';
  $('poreLegendDot').style.background=$('poreColor').value;
})();
