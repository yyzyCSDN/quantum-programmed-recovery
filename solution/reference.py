"""Raw quantum counting moments with marked programmed recovery and cohort transport."""
import bisect
import json
import math
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

PAIRS=[(a,k-a) for k in range(4) for a in range(k,-1,-1)]

def dec(x):
    z=np.asarray(x,float);return z[...,0]+1j*z[...,1]

def enc(x):return np.stack((x.real,x.imag),axis=-1).tolist()

def operators(seg,d):
    n=d*d;m=3*n;eye=np.eye(d);I=np.eye(n)
    E=[dec(j) for j in seg['emission']];B=[dec(j) for j in seg['bath']]
    loss=sum((j.conj().T@j for j in E+B),np.zeros((d,d),complex))
    emit=sum((np.kron(j.conj(),j) for j in E),np.zeros((n,n),complex))
    bath=sum((np.kron(j.conj(),j) for j in B),np.zeros((n,n),complex))
    local=[]
    for h in dec(seg['H']):
        local.append(-1j*(np.kron(eye,h)-np.kron(h.T,eye))-.5*(np.kron(eye,loss)+np.kron(loss.T,eye))+bath)
    L=np.zeros((m,m),complex);Jn=np.zeros_like(L);Jc=np.zeros_like(L)
    for q in range(3):L[q*n:(q+1)*n,q*n:(q+1)*n]=local[q]
    Jn[:n,:n]=(1-seg['eta'])*emit;Jn[n:2*n,:n]=seg['eta']*emit
    Jn[n:2*n,n:2*n]=emit;Jn[2*n:,2*n:]=emit
    L+=Jn
    L[:n,:n]-=seg['dark']*I;L[n:2*n,:n]+=seg['dark']*I
    L[n:2*n,n:2*n]-=seg['bandwidth']*I
    jumps=[seg['bandwidth']*np.kron(dec(b['K']).conj(),dec(b['K'])) for b in seg['readout']]
    jump=sum(jumps,np.zeros((n,n),complex))
    Jc[2*n:,n:2*n]=jump;L+=Jc
    A=np.zeros((10*m,10*m),complex)
    for k,(a,b) in enumerate(PAIRS):
        A[k*m:(k+1)*m,k*m:(k+1)*m]=L
        for i in range(a):
            j=PAIRS.index((i,b));A[k*m:(k+1)*m,j*m:(j+1)*m]+=math.comb(a,i)*Jn
        for i in range(b):
            j=PAIRS.index((a,i));A[k*m:(k+1)*m,j*m:(j+1)*m]+=math.comb(b,i)*Jc
    dead=np.zeros((4*n,4*n),complex)
    for a in range(4):
        dead[a*n:(a+1)*n,a*n:(a+1)*n]=local[2]+emit
        for i in range(a):dead[a*n:(a+1)*n,i*n:(i+1)*n]=math.comb(a,i)*emit
    return A,L-Jc,dead,jump,jumps

def solve(data,rtol=3e-11,atol=3e-13,mutation=None):
    initial=dec(data['rho0']);d=initial.shape[1];n=d*d;m=3*n
    release=float(data['initial_dead_remaining'])
    ends=np.cumsum([s['duration'] for s in data['segments']]);T=float(ends[-1]);starts=np.r_[0.,ends[:-1]]
    ops=[operators(s,d) for s in data['segments']]
    y=np.zeros(11*m+n+2,complex)
    y[:m]=np.concatenate([r.ravel(order='F') for r in initial]);y[10*m:11*m]=y[:m]
    # Each column holds a fixed C power, each row-block an N power.
    def columnize(v):
        out=np.zeros((4*n,4),complex)
        for k,(a,b) in enumerate(PAIRS):out[a*n:(a+1)*n,b]=v[k]
        return out
    def uncolumnize(v):return np.array([v[a*n:(a+1)*n,b] for a,b in PAIRS])
    # Interpolate entire analytic matrix exponentials, not the physical trajectory.
    # Each table is independently checked against direct expm before use.
    tables=[]
    for seg,op in zip(data['segments'],ops):
        D=op[2];duration=float(seg['duration'])
        for degree in [48,80,128]:
            nodes=np.cos(np.arange(degree+1)*np.pi/degree)
            weights=(-1.)**np.arange(degree+1);weights[[0,-1]]*=.5
            values=np.array([expm(D*(x+1)*duration/2) for x in nodes])
            flat=values.reshape(degree+1,-1)
            def evaluate(dt,nodes=nodes,weights=weights,flat=flat,duration=duration):
                delta=(2*dt/duration-1)-nodes;closest=np.argmin(abs(delta))
                if abs(delta[closest])<4e-15:return flat[closest].reshape(4*n,4*n)
                coeff=weights/delta;coeff/=sum(coeff)
                return (coeff@flat).reshape(4*n,4*n)
            if all(np.max(abs(evaluate(duration*f)-expm(D*duration*f)))<2e-12*max(1,np.max(abs(expm(D*duration*f)))) for f in [.071,.237,.519,.883]):break
        else:raise RuntimeError('Cohort propagator interpolation did not converge')
        tables.append(evaluate)
    def transport(v,s,t):
        if mutation=='freeze_dead_quantum':return v.copy()
        for i,(left,right) in enumerate(zip(starts,ends)):
            length=min(t,right)-max(s,left)
            if length>1e-14:v=tables[i](length)@v
        return v
    initial_dead=np.zeros((4*n,4),complex);initial_dead[:n,0]=initial[2].ravel(order='F')
    initial_return=uncolumnize(transport(initial_dead,0.,release)) if release<=T else None
    # Each source window has its own strictly increasing return map. Windows
    # from different branches/segments can overlap or reverse cohort order.
    windows=[];minimum=1.5
    for j,(left,right,seg) in enumerate(zip(starts,ends,data['segments'])):
        for b,branch in enumerate(seg['readout']):
            slope=branch['dead_slope'];base=branch['dead_base']
            rate=1+slope;offset=base-slope*left
            minimum=min(minimum,base,base+slope*(right-left))
            windows.append((j,b,left,right,rate,offset))
    boundaries={0.,T,release,*ends[:-1]}
    boundaries.update(np.arange(0.,T,minimum*.45).tolist())
    for _,_,left,right,rate,offset in windows:
        boundaries.update([rate*left+offset,rate*right+offset])
        if left<release<right:boundaries.add(rate*release+offset)
    cuts=sorted(x for x in boundaries if 0<=x<=T)
    history=[];history_starts=[];anchors={0.:y.copy()}
    def past(t):
        if t in anchors:return anchors[t]
        if t>T and t-T<1e-12 and T in anchors:return anchors[T]
        i=bisect.bisect_right(history_starts,t)-1
        if i<0:raise RuntimeError('missing past state')
        return history[i](t)
    for left,right in zip(cuts,cuts[1:]):
        if left==release and initial_return is not None:
            Y=y[:10*m].reshape(10,3,n);Y[:,0]+=initial_return;Y[:,2]-=initial_return
            Q=y[10*m:11*m].reshape(3,n);Q[0]+=initial_return[0];Q[2]-=initial_return[0]
        anchors[left]=y.copy()
        if right-left<1e-13:
            history.append(lambda t,value=y.copy():value.copy());history_starts.append(left)
            continue
        mid=(left+right)/2;idx=min(int(np.searchsorted(ends,mid)),len(ends)-1)
        A,Qop,_,jump,_=ops[idx]
        active=[w for w in windows if w[4]*w[2]+w[5]<mid<w[4]*w[3]+w[5]]
        if mutation=='one_return_window':active=active[:1]
        def rhs(t,z):
            dz=np.zeros_like(z);Y=z[:10*m];Q=z[10*m:11*m]
            dz[:10*m]=A@Y;dz[10*m:11*m]=Qop@Q
            instrument=jump@Q[n:2*n];flux=instrument.reshape(d,d,order='F').trace()
            dz[11*m:11*m+n]=instrument;dz[-2:]=[t*flux,t*t*flux]
            for source_idx,branch,left_s,right_s,rate,offset in active:
                s=min(right_s,max(left_s,(t-offset)/rate))
                old=past(s)[:10*m].reshape(10,3,n)
                source_jump=ops[source_idx][4][branch]
                if mutation=='forget_mark':
                    K=dec(data['segments'][source_idx]['readout'][branch]['K'])
                    source_jump=ops[source_idx][3]*(np.trace(K.conj().T@K).real/d)
                birth=[]
                for a,b in PAIRS:
                    value=sum((math.comb(b,j)*old[PAIRS.index((a,j)),1] for j in range(b+1)),np.zeros(n,complex))
                    birth.append(source_jump@value)
                jacobian=1. if mutation=='omit_jacobian' else 1./rate
                returning=jacobian*uncolumnize(transport(columnize(birth),s,t))
                D=dz[:10*m].reshape(10,3,n);D[:,0]+=returning;D[:,2]-=returning
            return dz
        sol=solve_ivp(rhs,(left,right),y,method='DOP853',rtol=rtol,atol=atol,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        history.append(sol.sol);history_starts.append(left);y=sol.y[:,-1]
    if T==release and initial_return is not None:
        Y=y[:10*m].reshape(10,3,n);Y[:,0]+=initial_return;Y[:,2]-=initial_return
        Q=y[10*m:11*m].reshape(3,n);Q[0]+=initial_return[0];Q[2]-=initial_return[0]
    anchors[T]=y.copy()
    reports=[]
    unpack=lambda z:np.array([row.reshape(d,d,order='F') for row in z.reshape(3,n)])
    trace=lambda z:np.trace(unpack(z),axis1=1,axis2=2).sum().real
    for t in data['times']:
        z=past(t);mom={ab:trace(z[k*m:(k+1)*m]) for k,ab in enumerate(PAIRS)}
        u,v=mom[1,0],mom[0,1];aa,ab,bb=mom[2,0],mom[1,1],mom[0,2]
        third=[mom[3,0]-3*u*aa+2*u**3,mom[2,1]-v*aa-2*u*ab+2*u*u*v,
               mom[1,2]-u*bb-2*v*ab+2*u*v*v,mom[0,3]-3*v*bb+2*v**3]
        reports.append({'rho':enc(unpack(z[:m])),'mean':[u,v],
            'covariance':[[aa-u*u,ab-u*v],[ab-u*v,bb-v*v]],'third':third,
            'no_click':enc(unpack(z[10*m:11*m])),
            'first_click':{'state':enc(z[11*m:11*m+n].reshape(d,d,order='F')),'time1':float(z[-2].real),'time2':float(z[-1].real)}})
    return {'reports':reports}

if __name__=='__main__':
    with open(sys.argv[1],encoding='utf-8') as f:data=json.load(f)
    with open(sys.argv[2],'w',encoding='utf-8') as f:json.dump(solve(data),f,allow_nan=False)
