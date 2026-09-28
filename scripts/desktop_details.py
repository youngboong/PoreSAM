"""Independent native Pore Details window, linked to its owning editor."""
import json
import threading


class DesktopDetails:
    def __init__(self):
        self._main=None
        self._details=None
        self._url=None
        self._title='Pore Details'
        self._lock=threading.RLock()

    def open_details(self):
        import webview
        with self._lock:
            if self._details is not None:
                self._details.restore();self._details.show()
                return True
            child=webview.create_window(self._title,self._url+'/pore-details-window',
                                        width=1000,height=760,min_size=(500,380),js_api=self,
                                        background_color='#ffffff')
            self._details=child
        def closed():
            with self._lock:
                if self._details is child:self._details=None
        child.events.closed+=closed
        return True

    def close_details(self):
        with self._lock:
            child=self._details;self._details=None
        if child is not None:child.destroy()

    def details_state(self,known_key=None):
        return self._main.evaluate_js('''(()=>{
          const data=window.detailsSnapshot??null;
          const stale=!!data&&(typeof state==='undefined'||!state||state.dataset!==data.dataset||state.revision!==data.revision);
          const key=data?data.dataset+':'+data.revision:null;
          return {key,stale,busy:stale||(typeof busy!=='undefined'&&busy),
            selected:window.getSelectedPores?.()??[],color:window.selectionColor?.()??'#ff00ff',
            data:key!==KNOWN&&data?{dataset:data.dataset,revision:data.revision,candidates:data.candidates,stats:data.stats}:null};
        })()'''.replace('KNOWN',json.dumps(known_key)))

    def select_details_pore(self,dataset,revision,candidate_id,toggle=False):
        return self._main.evaluate_js('''(()=>{
          const [dataset,revision,id,toggle]=ARGS;
          if(!state||busy||state.dataset!==dataset||state.revision!==revision)return false;
          if(id===null)window.clearPoreHighlight?.();else window.selectPoreFromDetails?.(id,toggle);
          return true;
        })()'''.replace('ARGS',json.dumps([dataset,revision,candidate_id,bool(toggle)])))

    def details_key(self,dataset,revision,key):
        if key not in ['Delete','Undo']:return False
        return self._main.evaluate_js('''(()=>{
          const [dataset,revision,key]=ARGS;
          if(!state||busy||state.dataset!==dataset||state.revision!==revision)return false;
          document.dispatchEvent(new KeyboardEvent('keydown',{key:key==='Undo'?'z':'Delete',code:key==='Undo'?'KeyZ':'Delete',ctrlKey:key==='Undo',bubbles:true}));
          return true;
        })()'''.replace('ARGS',json.dumps([dataset,revision,key])))
