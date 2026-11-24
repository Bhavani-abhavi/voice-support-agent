import json, sys
from fastapi.testclient import TestClient
from app.main import create_app
from app.config import Settings
app=create_app(Settings(llm_provider='mock'))
msgs = sys.argv[1:] or ["Hi, I think I was charged twice this month. Can I get a refund? I'm alex.rivera@example.com, ZIP 64111.","What happens to my notes if my payment fails?","Can you help me pick a good laptop for college?"]
sim = 'none'
with TestClient(app) as c:
    sid=c.post('/api/sessions').json()['session_id']
    for msg in msgs:
        if msg.startswith('SIM='): sim=msg[4:]; continue
        r=c.post('/api/chat',json={'session_id':sid,'message':msg,'simulate':sim})
        print('====',msg[:60], r.status_code)
        toks=[]
        for block in r.text.split('\n\n'):
            lines=block.strip().split('\n')
            if len(lines)<2: continue
            ev=lines[0][7:]; d=json.loads(lines[1][6:])
            if ev=='token': toks.append(d['text']); continue
            if ev=='trace': print('trace', {k:d[k] for k in ('outcome','latency_ms','ttft_ms','input_tokens','output_tokens','cost_usd')}, 'spans', [(s['name'],s['status']) for s in d['spans']]); continue
            print(ev, json.dumps(d)[:220])
        print('TOKENS:', ''.join(toks)[:300])
