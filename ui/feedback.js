(() => {
  const get=id=>document.getElementById(id),dialog=get('feedbackDialog');
  let attachments=[],sending=false,attaching=false,identifier=crypto.randomUUID();
  const MAX_IMAGE=5*1024*1024,MAX_TOTAL=8*1024*1024;
  function status(message,error=false){get('feedbackStatus').textContent=message;get('feedbackStatus').classList.toggle('error',error)}
  function controls(){
    const waiting=sending||attaching;
    get('openFeedback').disabled=waiting;
    for(const id of ['feedbackMessage','feedbackFiles','attachFeedbackImage','closeFeedback'])get(id).disabled=waiting;
    get('sendFeedback').disabled=waiting||(!get('feedbackMessage').value.trim()&&!attachments.length);
    get('sendFeedback').textContent=sending?'Sending...':'Send';
    dialog.querySelectorAll('[data-feedback-remove]').forEach(button=>button.disabled=waiting);
  }
  function render(){
    get('feedbackImages').replaceChildren();
    attachments.forEach((item,index)=>{
      const card=document.createElement('div');card.className='feedback-image';
      const image=document.createElement('img');image.src=item.url;image.alt=item.name;
      const remove=document.createElement('button');remove.type='button';remove.textContent='×';remove.dataset.feedbackRemove='';remove.setAttribute('aria-label','Remove '+item.name);
      remove.onclick=()=>{attachments.splice(index,1);identifier=crypto.randomUUID();render();status('')};
      card.append(image,remove);get('feedbackImages').append(card);
    });controls();
  }
  async function attach(files){
    if(sending||attaching)return;attaching=true;controls();
    try{
      const added=[];let total=attachments.reduce((sum,item)=>sum+item.size,0);
      for(const file of files){
        if(!['image/png','image/jpeg','image/webp','image/gif'].includes(file.type))throw new Error('Use PNG, JPG, WebP or GIF images.');
        if(attachments.length+added.length>=4)throw new Error('Attach up to four images.');
        total+=file.size;if(!file.size||file.size>MAX_IMAGE||total>MAX_TOTAL)throw new Error('Images must be under 5 MB each and 8 MB in total.');
        const url=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('Cannot read the image.'));reader.readAsDataURL(file)});
        added.push({name:file.name||'Pasted image',size:file.size,url});
      }
      attachments.push(...added);identifier=crypto.randomUUID();render();status('');
    }catch(error){status(error.message,true)}finally{attaching=false;controls()}
  }
  get('openFeedback').onclick=()=>{dialog.showModal();status('');controls();get('feedbackMessage').focus()};
  get('closeFeedback').onclick=()=>dialog.close();
  dialog.addEventListener('cancel',event=>{if(sending)event.preventDefault()});
  get('feedbackMessage').oninput=()=>{identifier=crypto.randomUUID();controls()};
  get('attachFeedbackImage').onclick=()=>get('feedbackFiles').click();
  get('feedbackFiles').onchange=async()=>{await attach([...get('feedbackFiles').files]);get('feedbackFiles').value=''};
  dialog.addEventListener('paste',event=>{
    if(sending)return;
    const files=[...(event.clipboardData?.items||[])].filter(item=>item.kind==='file'&&item.type.startsWith('image/')).map(item=>item.getAsFile()).filter(Boolean);
    if(files.length){event.preventDefault();attach(files)}
  });
  get('sendFeedback').onclick=async()=>{
    if(sending)return;sending=true;controls();status('Sending...');
    try{
      const response=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json','X-Pore-Editor':'1'},body:JSON.stringify({id:identifier,message:get('feedbackMessage').value,images:attachments.map(item=>({data:item.url.split(',')[1]}))})});
      const result=await response.json();if(!response.ok||!result.sent)throw new Error(result.error||'Could not send feedback.');
      attachments=[];get('feedbackMessage').value='';identifier=crypto.randomUUID();render();status('Thank you. Your feedback has been sent.');
    }catch(error){status(error.message,true)}finally{sending=false;controls()}
  };
  const previous=window.updateWorkspaceControls;
  window.updateWorkspaceControls=()=>{previous?.();controls()};
  controls();
})();
