/* Large-pore discovery shares the editor's preview, save and undo workflow. */
(()=>{
  let dataset=null,key=null,overview=null,picking=false,revision=null;
  function pickingMode(enabled){
    picking=enabled;
    $('largePick').classList.toggle('active',enabled);
    $('largePick').textContent=enabled?'이미지 선택 중 · 눌러서 종료':'이미지에서 후보 선택';
  }
  window.stopLargePicking=()=>{pickingMode(false);$('largeOutline').checked=false};
  window.onLargeState=data=>{
    if(dataset===data.dataset&&revision===data.revision)return;
    revision=data.revision;
    $('largeReplace').checked=false;
    dataset=data.dataset;key=null;overview=null;pickingMode(false);
    $('largeReview').classList.add('hidden');$('largeStatus').textContent='';
    $('largeOutline').checked=false;draw();
  };
  window.drawLargeOverview=context=>{
    if(state?.dataset===dataset&&overview&&$('largeOutline').checked)context.drawImage(overview,0,0);
  };
  window.hasLargeOverview=()=>state?.dataset===dataset&&overview&&$('largeOutline').checked;
  async function refreshOverview(){
    const data=await api('large-overview',{...payload(),key,allow_replacement:$('largeReplace').checked});
    overview=await image(data.image);
    $('largeCandidate').replaceChildren(new Option('추가할 pore 선택',''));
    data.candidates.forEach((c,i)=>$('largeCandidate').add(new Option((i+1)+'번 · '+c.area_um2.toFixed(1)+' µm²'+(c.alternatives>1?' · 경계 '+c.alternatives+'개':'')+(c.replacement_ids.length?' · 기존 교체':''),c.id)));
    $('largeReview').classList.remove('hidden');pickingMode(data.candidates.length>0);
    $('largeOutline').checked=true;
    $('largeStatus').textContent='기존 pore '+state.candidates.length+'개 · 겹치지 않는 검토 후보 '+data.candidates.length+'개. 경계 대안은 선택할 때만 표시합니다.';
    draw();
  }
  $('largeReplace').onchange=()=>work('기존 pore와 겹침을 다시 확인합니다.',async()=>{clear();await refreshOverview()});
  async function pick(selection){
    clear();$('target').value='';
    await presentPreview(await api('large-pick',{...payload(),key,allow_replacement:$('largeReplace').checked,...selection}));
    $('largeOutline').checked=false;draw();
    setStatus('미리보기 후보를 비교하고 원하는 경계를 적용하세요. 아직 저장되지 않았습니다.');
  }
  window.handleLargePointer=point=>{
    if(!picking||state?.dataset!==dataset||!key)return false;
    work('이 위치의 큰 pore 후보를 확인합니다.',()=>pick({point}));return true;
  };
  $('largePick').onclick=()=>{pickingMode(!picking);setStatus(picking?'찾으려는 큰 pore 내부를 클릭하세요.':'박스나 타원형으로 영역을 지정할 수 있습니다.')};
  $('largeOutline').onchange=draw;
  $('largeStrength').onchange=()=>{
    key=null;overview=null;pickingMode(false);clear();
    $('largeReview').classList.add('hidden');
    $('largeStatus').textContent='변경한 강도로 큰 pore 추가 탐색을 실행하세요.';
  };
  $('largeCandidate').onchange=()=>{
    const id=Number($('largeCandidate').value);
    if(id)work('큰 pore 후보를 표시합니다.',()=>pick({candidate_id:id}));
  };
  $('largeSearch').onclick=()=>{
    if(!state)return;
    work('큰 pore를 추가로 찾고 있습니다. 첫 탐색에는 시간이 걸립니다.',async()=>{
      clear();dataset=state.dataset;key=null;overview=null;pickingMode(false);
      $('largeReview').classList.add('hidden');$('largeProgress').classList.remove('hidden');
      $('largeProgress').value=0;
      try{
        const started=await api('large-search',{...payload(),strength:$('largeStrength').value});
        let job;
        do{
          job=await api('large-job',started);
          $('largeProgress').value=job.progress||0;$('largeStatus').textContent=job.message;
          if(job.status==='failed')throw new Error(job.message);
          if(job.status!=='complete')await new Promise(resolve=>setTimeout(resolve,1000));
        }while(job.status!=='complete');
        key=job.key;
        await refreshOverview();
        setStatus('파란색은 기존 pore, 노란색은 겹치지 않는 검토 후보입니다. 추가할 pore를 선택하세요.');
      }catch(error){$('largeStatus').textContent=error.message;throw error}
      finally{$('largeProgress').classList.add('hidden')}
    });
  };
})();
