import { readFileSync } from 'fs';
function makeEl(){
  return { innerHTML:'', textContent:'', value:'', checked:true, style:{}, dataset:{},
    classList:{ add(){}, remove(){}, toggle(){}, contains(){return false} },
    addEventListener(){}, onclick:null, onchange:null, parentElement:null };
}
const els = {};
globalThis.document = {
  querySelector: s => els[s] || (els[s]=makeEl()),
  querySelectorAll: () => [],
  getElementById: id => els['#'+id] || (els['#'+id]=makeEl()),
  hidden:false,
};
globalThis.localStorage = { _d:{}, getItem(k){return this._d[k]??null}, setItem(k,v){this._d[k]=v}, removeItem(k){delete this._d[k]} };
globalThis.location = { reload(){} };
const _fetch = globalThis.fetch;
globalThis.fetch = (url, opts) => _fetch(url.startsWith('http') ? url : 'http://127.0.0.1:8000'+url, opts);

const html = readFileSync(new URL('../app/static/dashboard.html', import.meta.url), 'utf8');
const js = /<script>([\s\S]*?)<\/script>/.exec(html)[1].replace(/\/\* ============ boot[\s\S]*$/, '');
const driver = `
;(async()=>{
  const list = await api('GET','/jobs?limit=50');
  const states = {};
  let warns = 0, count = 0, evtCount = 0;
  for (const j of list.jobs.slice(0,12)) {
    const detail = await api('GET','/jobs/'+j.job_id);
    for (const it of detail.items) {
      count++;
      states[it.state] = (states[it.state]||0)+1;
      const card = itemCard(it);
      for (const bad of ['undefined','NaN','[object Object]'])
        if (card.includes(bad)) { warns++; if (warns<5) console.log('WARN', bad, '|', it.smart_ad_id, '| state', it.state); }
    }
    const ev = await api('GET','/jobs/'+j.job_id+'/events?limit=200');
    selEvents = ev.events||[]; selDetail = detail;
    renderDetail();
    evtCount += selEvents.length;
  }
  console.log('rendered', count, 'items across states:', JSON.stringify(states));
  console.log('events rendered:', evtCount, '| warnings:', warns);
  console.log(warns===0 ? 'DEEP-RENDER-OK' : 'DEEP-RENDER-WARNINGS');
})().catch(e=>{ console.error('FAIL:', e.message); process.exit(1); });
`;
const line = readFileSync(new URL('../.env', import.meta.url), 'utf8').split(/\r?\n/).find(l=>l.startsWith('API_KEYS='));
const [cid, sec] = line.slice(9).trim().split(',')[0].split(':');
eval(js + '\n' + `store.save(${JSON.stringify(cid)}, ${JSON.stringify(sec)});` + driver);
