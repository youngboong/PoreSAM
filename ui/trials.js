// Keep exploratory comparisons separate from the currently selected pore measurements.
(async()=>{
  try{
    const response=await fetch('/api/trials');if(!response.ok)return;
    const data=await response.json();if(!data.trials.length)return;
    const details=document.createElement('details');details.className='panel';details.id='trialReview';details.style.marginTop='20px';
    const summary=document.createElement('summary');summary.textContent='PI · GF 방식 비교 실험 보기';summary.style.cursor='pointer';summary.style.fontWeight='bold';
    const note=document.createElement('p');note.textContent='임시 외곽 기준으로 비교한 검토 전 실험입니다. 현재 이미지의 자동 결과나 통계에는 적용되지 않습니다.';note.className='hint';
    const select=document.createElement('select');select.setAttribute('aria-label','비교 실험 선택');
    data.trials.forEach(t=>select.add(new Option(t.name,t.url)));
    const frame=document.createElement('iframe');frame.id='trialFrame';frame.title='PI와 GF 분할 방식 비교';frame.style.cssText='display:block;width:100%;height:1050px;border:0;margin-top:15px;background:white';
    details.ontoggle=()=>{if(details.open&&!frame.getAttribute('src'))frame.src=select.value};
    select.onchange=()=>{frame.src=select.value};
    details.append(summary,note,select,frame);$('analysisPanel').append(details);
  }catch(error){console.warn('Trial comparison unavailable:',error.message)}
})();
