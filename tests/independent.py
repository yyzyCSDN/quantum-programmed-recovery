"""Independent event-graph factorial-moment DDE and direct cohort evolution."""
import json
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

PAIRS=[(a,b) for b in range(4) for a in range(4-b)]

def dec(value):
    a=np.asarray(value,float);return a[...,0]+1j*a[...,1]

def enc(value):return np.stack((value.real,value.imag),axis=-1).tolist()

def solve(data,rtol=2e-10,atol=2e-12):
    initial=dec(data['rho0']);d=initial.shape[1];n=d*d;m=3*n
    remaining=data['initial_dead_remaining'];total=sum(s['duration'] for s in data['segments'])
    edges=np.r_[0.,np.cumsum([s['duration'] for s in data['segments']])]
    specs=[]
    for seg in data['segments']:
        H=dec(seg['H']);events=[];eye=np.eye(d)
        emissions=[dec(op) for op in seg['emission']];baths=[dec(op) for op in seg['bath']]
        for q in range(3):
            for op in baths:events.append((q,q,op,0,0))
            for op in emissions:
                if q==0:
                    events.extend([(q,0,np.sqrt(1-seg['eta'])*op,1,0),(q,1,np.sqrt(seg['eta'])*op,1,0)])
                else:events.append((q,q,op,1,0))
        events.append((0,1,np.sqrt(seg['dark'])*eye,0,0))
        events.extend((1,2,np.sqrt(seg['bandwidth'])*dec(branch['K']),0,1) for branch in seg['readout'])
        G=-1j*H
        for q,r,J,N,C in events:G[q]-=.5*J.conj().T@J
        def dead_rhs(v):
            R=v.reshape(4,d,d);D=G[2]@R+R@G[2].conj().T
            for op in emissions+baths:D+=op@R@op.conj().T
            for a in range(1,4):
                for op in emissions:D[a]+=a*(op@R[a-1]@op.conj().T)
            return D.ravel()
        D=np.column_stack([dead_rhs(col) for col in np.eye(4*n,dtype=complex)])
        specs.append((events,G,D,seg['bandwidth'],[dec(branch['K']) for branch in seg['readout']]))
    cache={}
    def cohort(R,s,t):
        Z=np.zeros((4*n,4),complex)
        for k,(a,b) in enumerate(PAIRS):Z[a*n:(a+1)*n,b]=R[k].ravel()
        for j in range(len(specs)):
            dt=max(0.,min(t,edges[j+1])-max(s,edges[j]))
            if dt<1e-14:continue
            key=(j,round(float(dt),13))
            if key not in cache:
                A=specs[j][2]*dt
                cache[key]=expm_multiply(A,np.eye(4*n,dtype=complex),traceA=np.trace(A))
            Z=cache[key]@Z
        return np.array([Z[a*n:(a+1)*n,b].reshape(d,d) for a,b in PAIRS])
    dead_initial=np.zeros((10,d,d),complex);dead_initial[0]=initial[2]
    pulse=cohort(dead_initial,0.,remaining) if remaining<=total else None
    y=np.zeros(11*m+n+2,complex);y[:m]=initial.ravel();y[10*m:11*m]=initial.ravel()
    branches=[];delay_bound=1.5
    nodes={0.,float(total),float(remaining),*edges[1:-1]}
    for j,seg in enumerate(data['segments']):
        for k,b in enumerate(seg['readout']):
            slope=b['dead_slope'];base=b['dead_base']
            first=edges[j]+base
            last=edges[j+1]+base+slope*seg['duration']
            branches.append((j,k,first,last,1+slope))
            nodes.update([first,last])
            delay_bound=min(delay_bound,base,base+slope*seg['duration'])
            if edges[j]<remaining<edges[j+1]:nodes.add(remaining+base+slope*(remaining-edges[j]))
    nodes.update(np.arange(0.,total,delay_bound/3).tolist())
    nodes=sorted(t for t in nodes if 0<=t<=total)
    past_segments=[];values={0.:y.copy()}
    def lookup(t):
        if t in values:return values[t]
        if t>total and t-total<1e-12 and total in values:return values[total]
        for begin,end,solution in reversed(past_segments):
            if begin<=t<=end:return solution(t)
        raise RuntimeError('No historical state for '+str(t))
    for begin,end in zip(nodes,nodes[1:]):
        if begin==remaining and pulse is not None:
            R=y[:11*m].reshape(11,3,d,d);R[:10,0]+=pulse;R[:10,2]-=pulse
            R[10,0]+=pulse[0];R[10,2]-=pulse[0]
        values[begin]=y.copy()
        if end-begin<1e-13:
            past_segments.append((begin,end,lambda t,value=y.copy():value.copy()))
            continue
        middle=(begin+end)/2
        j=min(int(np.searchsorted(edges[1:],middle)),len(specs)-1)
        events,G,_,_,_=specs[j]
        active=[b for b in branches if b[2]<middle<b[3]]
        def rhs(t,z):
            R=z[:11*m].reshape(11,3,d,d)
            D=G[None,...]@R+R@G.conj().transpose(0,2,1)[None,...]
            F=np.zeros((d,d),complex)
            for q,r,J,N,C in events:
                gain=J@R[:,q]@J.conj().T;D[:10,r]+=gain[:10]
                if not C:D[10,r]+=gain[10]
                else:F+=gain[10]
                for k,(a,b) in enumerate(PAIRS):
                    if N and a:D[k,r]+=a*gain[PAIRS.index((a-1,b))]
                    if C and b:D[k,r]+=b*gain[PAIRS.index((a,b-1))]
            for source,mark,first,last,stretch in active:
                source_time=min(edges[source+1],max(edges[source],edges[source]+(t-first)/stretch))
                old=lookup(source_time)[:10*m].reshape(10,3,d,d)[:,1]
                _,_,_,past_rate,matrices=specs[source];K=matrices[mark]
                born=np.empty_like(old)
                for k,(a,b) in enumerate(PAIRS):
                    weight=old[k]+(b*old[PAIRS.index((a,b-1))] if b else 0)
                    born[k]=(past_rate/stretch)*(K@weight@K.conj().T)
                returns=cohort(born,source_time,t);D[:10,0]+=returns;D[:10,2]-=returns
            flux=F.trace()
            return np.concatenate((D.ravel(),F.ravel(),[t*flux,t*t*flux]))
        integrated=solve_ivp(rhs,(begin,end),y,rtol=rtol,atol=atol,method='RK45',dense_output=True)
        if not integrated.success:raise RuntimeError(integrated.message)
        y=integrated.y[:,-1];past_segments.append((begin,end,integrated.sol))
    if total==remaining and pulse is not None:
        R=y[:11*m].reshape(11,3,d,d);R[:10,0]+=pulse;R[:10,2]-=pulse
        R[10,0]+=pulse[0];R[10,2]-=pulse[0]
    values[total]=y.copy();output=[]
    for t in data['times']:
        z=lookup(t);R=z[:11*m].reshape(11,3,d,d)
        f={ab:np.trace(R[k],axis1=1,axis2=2).sum().real for k,ab in enumerate(PAIRS)}
        x,c=f[1,0],f[0,1];xx,cc,xc=f[2,0]+x,f[0,2]+c,f[1,1]
        third=[f[3,0]+3*f[2,0]+x-3*x*xx+2*x**3,f[2,1]+xc-c*xx-2*x*xc+2*x*x*c,
               f[1,2]+xc-x*cc-2*c*xc+2*x*c*c,f[0,3]+3*f[0,2]+c-3*c*cc+2*c**3]
        output.append({'rho':enc(R[0]),'mean':[x,c],'covariance':[[xx-x*x,xc-x*c],[xc-x*c,cc-c*c]],
            'third':third,'no_click':enc(R[10]),'first_click':{'state':enc(z[11*m:11*m+n].reshape(d,d)),
            'time1':float(z[-2].real),'time2':float(z[-1].real)}})
    return {'reports':output}
