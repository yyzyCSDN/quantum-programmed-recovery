"""Fluorescence protocol report command line interface."""
import json
import sys
import numpy as np
from scipy.integrate import solve_ivp

def matrix(value):
    a=np.asarray(value,float)
    return a[...,0]+1j*a[...,1]

def packed(value):
    return np.stack((value.real,value.imag),axis=-1).tolist()

def evolve(data):
    initial=matrix(data['rho0']); d=initial.shape[1]; m=3*d*d
    y=np.zeros(m+4,complex); y[:m]=initial.ravel()
    history=[]; start=0.
    for seg in data['segments']:
        H=matrix(seg['H']); instruments=[matrix(b['K']) for b in seg['readout']]
        recovery=1./np.mean([b['dead_base'] for b in seg['readout']])
        emissions=[matrix(j) for j in seg['emission']]
        baths=[matrix(j) for j in seg['bath']]
        loss=sum((j.conj().T@j for j in emissions+baths),np.zeros((d,d),complex))
        def rhs(t,y):
            r=y[:m].reshape(3,d,d)
            dr=np.zeros_like(r); emitted=0.; accepted=0.
            for q in range(3):
                dr[q]=-1j*(H[q]@r[q]-r[q]@H[q])-.5*(loss@r[q]+r[q]@loss)
                for j in baths: dr[q]+=j@r[q]@j.conj().T
            for q in range(3):
                for j in emissions:
                    gain=j@r[q]@j.conj().T
                    emitted+=np.trace(gain).real
                    if q==0:
                        dr[0]+=(1-seg['eta'])*gain
                        dr[1]+=seg['eta']*gain
                        accepted+=seg['eta']*np.trace(gain).real
                    else: dr[q]+=gain
            dr[0]-=seg['dark']*r[0]
            dr[1]+=seg['dark']*r[0]-seg['bandwidth']*r[1]
            dr[2]+=seg['bandwidth']*sum((K@r[1]@K.conj().T for K in instruments),np.zeros((d,d),complex))-recovery*r[2]
            dr[0]+=recovery*r[2]
            accepted+=seg['dark']*np.trace(r[0]).real
            survival=np.exp(-y[m+1].real)
            return np.concatenate((dr.ravel(),[emitted,accepted,t*accepted*survival,t*t*accepted*survival]))
        end=start+seg['duration']
        result=solve_ivp(rhs,(start,end),y,method='DOP853',rtol=1e-9,atol=1e-11,dense_output=True)
        if not result.success: raise RuntimeError(result.message)
        history.append((end,result.sol)); y=result.y[:,-1]; start=end
    return history,d

def solve(data):
    history,d=evolve(data); m=3*d*d; reports=[]
    for t in data['times']:
        sol=next(sol for end,sol in history if t<=end+1e-13)
        y=sol(t); rho=y[:m].reshape(3,d,d); n,c=y[m:m+2].real
        survival=np.exp(-c)
        reports.append({'rho':packed(rho),'mean':[float(n),float(c)],
                        'covariance':[[float(n),0.],[0.,float(c)]],
                        'third':[float(n),0.,0.,float(c)],
                        'no_click':packed(survival*rho),
                        'first_click':{'state':packed((1-survival)*rho.sum(axis=0)),
                                       'time1':float(y[-2].real),'time2':float(y[-1].real)}})
    return {'reports':reports}

if __name__=='__main__':
    with open(sys.argv[1],encoding='utf-8') as stream: data=json.load(stream)
    result=solve(data)
    with open(sys.argv[2],'w',encoding='utf-8') as stream: json.dump(result,stream,allow_nan=False)
