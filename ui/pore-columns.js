// The same column picker is used by the browser popup and native details window.
window.poreColumnCatalog=[
  ['candidate_id','ID','Saved pore identifier.','id'],
  ['length_um','Length (µm)','Major axis of the moment-equivalent ellipse.'],
  ['width_um','Width (µm)','Minor axis of the moment-equivalent ellipse.'],
  ['aspect_ratio','Aspect ratio','Length / Width.'],
  ['equivalent_diameter_um','Equivalent diameter (µm)','Diameter of a circle with the same area.'],
  ['roundness','Roundness','4 × Area / (π × Length²).'],
  ['complete','Complete','Yes when the mask does not touch the image boundary.','boolean'],
  ['image_area_percent','Area fraction (%)','Pore area / analysis image area × 100.'],
  ['angle_deg','Angle (°)','Major axis direction, clockwise from image X, 0–180°. Undefined for equal axes.'],
  ['area_um2','Area (µm²)','Visible pore mask area.'],
  ['area_box_ratio','Area/Box','Mask area / axis-aligned bounding box area, 0–1.'],
  ['brightness_max','Brightness max','Maximum source grayscale value inside the pore (0–255).'],
  ['brightness_mean','Brightness mean','Mean source grayscale value inside the pore (0–255).'],
  ['brightness_min','Brightness min','Minimum source grayscale value inside the pore (0–255).'],
  ['brightness_std','Brightness std dev','Population standard deviation of source grayscale values inside the pore.'],
  ['centroid_x_um','Center X (µm)','Geometric centroid X; image origin is the top-left pixel center.'],
  ['centroid_y_um','Center Y (µm)','Geometric centroid Y; positive downward.'],
  ['convexity','Convexity','Convex hull perimeter / filled exterior perimeter; Crofton estimate, clipped to 0–1.'],
  ['integral_density','Integral density','Sum of source grayscale values inside the pore (gray value × pixels).'],
  ['mass_center_x_um','Mass center X (µm)','Brightness-weighted centroid X, not physical mass. Undefined when total brightness is zero.'],
  ['mass_center_y_um','Mass center Y (µm)','Brightness-weighted centroid Y, not physical mass. Undefined when total brightness is zero.'],
  ['perimeter_um','Perimeter (µm)','Crofton perimeter estimate, including any interior boundaries.'],
  ['rectangle_bottom_um','Rectangle bottom (µm)','Bottom of axis-aligned bounding box; exclusive pixel bound × scale.'],
  ['rectangle_left_um','Rectangle left (µm)','Left of axis-aligned bounding box; minimum pixel index × scale.'],
  ['rectangle_right_um','Rectangle right (µm)','Right of axis-aligned bounding box; exclusive pixel bound × scale.'],
  ['rectangle_top_um','Rectangle top (µm)','Top of axis-aligned bounding box; minimum pixel index × scale.'],
  ['solidity','Solidity','Mask area / raster convex hull area.'],
  ['circularity','Circularity','4π × Area / Perimeter²; clipped to 0–1.']
];
window.createPoreColumnPicker=onChange=>{
  const catalog=window.poreColumnCatalog,byKey=new Map(catalog.map(item=>[item[0],item]));
  const defaults=catalog.slice(0,8).map(item=>item[0]),storageKey='poreVisibleColumns.v1';
  let columns=defaults.slice(),draft=[];
  try{const saved=JSON.parse(localStorage.getItem(storageKey));if(Array.isArray(saved)){const valid=[...new Set(saved)].filter(key=>byKey.has(key));if(valid.length)columns=valid}}catch{}
  const button=document.createElement('button');button.id='openDetailsColumns';button.type='button';button.className='icon-button';
  button.title='Columns';button.setAttribute('aria-label','Choose columns');
  button.innerHTML='<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16M15 4v16M3 9h18"/></svg>';
  $('downloadDetails').before(button);
  const dialog=document.createElement('dialog');dialog.id='poreColumnsDialog';dialog.setAttribute('aria-labelledby','columnsTitle');
  dialog.innerHTML=`<div class="columns-titlebar"><h2 id="columnsTitle">Columns</h2><button id="closeColumns" aria-label="Close columns">×</button></div>
    <div class="columns-body"><div class="columns-lists">
    <label>Available columns<select id="availableColumns" multiple size="18" aria-label="Available columns"></select></label>
    <div class="columns-actions"><button id="addColumns" aria-label="Show selected columns">&gt;&gt;</button><button id="removeColumns" aria-label="Hide selected columns">&lt;&lt;</button><button id="clearColumns">Clear</button><div class="columns-spacer"></div><button id="columnsUp">Up</button><button id="columnsDown">Down</button></div>
    <label>Visible columns<select id="visibleColumns" multiple size="18" aria-label="Visible columns"></select></label></div>
    <p id="columnDescription" class="hint" role="status">Select columns to show, then arrange their order.</p>
    <div class="columns-footer"><button id="resetColumns">Defaults</button><span></span><button id="confirmColumns" class="primary">OK</button><button id="cancelColumns">Cancel</button></div></div>`;
  document.body.append(dialog);
  const available=$('availableColumns'),visible=$('visibleColumns');
  const selection=list=>[...list.selectedOptions].map(option=>option.value);
  function controls(){
    if(!dialog.open)return;
    dialog.querySelectorAll('button,select').forEach(element=>element.disabled=false);
    $('addColumns').disabled=!available.selectedOptions.length;
    $('removeColumns').disabled=!visible.selectedOptions.length;
    $('clearColumns').disabled=!draft.length;
    $('confirmColumns').disabled=!draft.length;
    const picked=new Set(selection(visible));
    $('columnsUp').disabled=!draft.some((key,i)=>i>0&&picked.has(key)&&!picked.has(draft[i-1]));
    $('columnsDown').disabled=!draft.some((key,i)=>i<draft.length-1&&picked.has(key)&&!picked.has(draft[i+1]));
  }
  window.updateColumnControls=controls;
  function render(selected=[]){
    available.replaceChildren();visible.replaceChildren();
    for(const [key,label,help] of [...catalog].sort((a,b)=>a[1].localeCompare(b[1]))){
      if(draft.includes(key))continue;const option=new Option(label,key);option.title=help;available.add(option);
    }
    for(const key of draft){const item=byKey.get(key),option=new Option(item[1],key);option.title=item[2];option.selected=selected.includes(key);visible.add(option)}
    controls();
  }
  function describe(list){const key=selection(list).at(-1);$('columnDescription').textContent=byKey.get(key)?.[2]??'Select columns to show, then arrange their order.';controls()}
  available.onchange=()=>describe(available);visible.onchange=()=>describe(visible);
  function add(){const keys=selection(available);draft.push(...keys);render(keys)}
  function remove(){const keys=new Set(selection(visible));draft=draft.filter(key=>!keys.has(key));render()}
  $('addColumns').onclick=available.ondblclick=add;
  $('removeColumns').onclick=visible.ondblclick=remove;
  $('clearColumns').onclick=()=>{draft=[];render()};
  function move(direction){
    const selected=selection(visible),picked=new Set(selected);
    const indices=direction<0?[...draft.keys()]:[...draft.keys()].reverse();
    for(const i of indices){const j=i+direction;if(j>=0&&j<draft.length&&picked.has(draft[i])&&!picked.has(draft[j]))[draft[i],draft[j]]=[draft[j],draft[i]]}
    render(selected);
  }
  $('columnsUp').onclick=()=>move(-1);$('columnsDown').onclick=()=>move(1);
  $('resetColumns').onclick=()=>{draft=defaults.slice();render()};
  const close=()=>{dialog.close();button.focus()};
  $('closeColumns').onclick=$('cancelColumns').onclick=close;
  dialog.addEventListener('cancel',()=>button.focus());
  $('confirmColumns').onclick=()=>{
    if(!draft.length)return;columns=draft.slice();
    try{localStorage.setItem(storageKey,JSON.stringify(columns))}catch{}
    onChange();close();
  };
  button.onclick=()=>{draft=columns.slice();dialog.showModal();render();available.focus()};
  return ()=>columns.map(key=>byKey.get(key));
};
