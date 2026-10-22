import numpy as np
import hashlib
import json
from pathlib import Path

GOLDEN=json.loads((Path(__file__).parent/"expected.json").read_text())

def expected(data):
    key=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    return GOLDEN[key]["expected"]

def verify(data, actual, want=None):
    if want is None: want=expected(data)
    failures=[]
    def compare(a,b,path):
        if isinstance(b,dict):
            if not isinstance(a,dict): failures.append(path+': expected object'); return
            for key in b:
                if key not in a: failures.append(path+'.'+key+': missing')
                else: compare(a[key],b[key],path+'.'+key)
        elif path=='reports':
            if not isinstance(a,list) or len(a)!=len(b): failures.append('reports: wrong count'); return
            for k,(x,y) in enumerate(zip(a,b)): compare(x,y,'reports[%d]'%k)
        else:
            try:
                def numeric(v):
                    if isinstance(v,list): return all(numeric(item) for item in v)
                    return isinstance(v,(int,float)) and not isinstance(v,bool)
                if not numeric(a):
                    failures.append(path+': expected JSON numbers'); return
                x=np.asarray(a,dtype=float); y=np.asarray(b,dtype=float)
                atol,rtol=(2e-6,1e-5) if any(s in path for s in ['third','time1','time2']) else (2e-7,2e-6)
                if x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.allclose(x,y,atol=atol,rtol=rtol):
                    failures.append(path+': numerical mismatch')
            except (ValueError,TypeError): failures.append(path+': invalid numbers')
    if not isinstance(actual,dict) or 'reports' not in actual: return ['missing reports']
    compare(actual['reports'],want['reports'],'reports')
    return failures
