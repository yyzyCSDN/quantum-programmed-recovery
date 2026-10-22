"""Synthetic protocols for marked and time-programmed recovery."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

def enc(x):
    z=np.asarray(x,complex);return np.stack((z.real,z.imag),axis=-1).tolist()

def branch(K,base=.7,slope=0.):return {'K':enc(K),'dead_base':base,'dead_slope':slope}

def segment(d=2,duration=1.,**kw):
    s={'duration':duration,'H':enc(np.zeros((3,d,d))),'emission':[],'bath':[],
       'eta':.7,'dark':0.,'bandwidth':1.,'readout':[branch(np.eye(d))]};s.update(kw);return s

def record(segments,rho=None,remaining=.4,times=None):
    d=len(segments[0]['H'][0]);T=sum(s['duration'] for s in segments)
    if rho is None:rho=np.zeros((3,d,d),complex);rho[0,0,0]=1
    return {'rho0':enc(rho),'segments':segments,'initial_dead_remaining':remaining,
            'times':times if times is not None else [T,0.,.37*T,T,.8*T]}

def generate():
    rows=[]
    def put(name,group,data):rows.append((name,group,data))
    E=np.array([[0,1],[0,0]],complex);X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.])
    h=np.array([[.2,.7+.4j],[.7-.4j,-.2]])
    psi=np.array([1.,1j])/np.sqrt(2);coherent=np.outer(psi,psi.conj())
    excited=np.zeros((3,2,2),complex);excited[0,1,1]=1
    pending=np.zeros_like(excited);pending[1]=coherent
    dead=np.zeros_like(excited);dead[2]=coherent
    P0=np.diag([1.,0.]);P1=np.diag([0.,1.])
    weak=[expm(-.4j*X)@np.diag(np.sqrt([.8,.2])),expm(-.7j*Z)@np.diag(np.sqrt([.2,.8]))]
    put('quiet_ready','regression',record([segment()]))
    put('initial_recovery_front','memory',record([segment(H=enc([h,-h,.6*X]))],dead,remaining=.5,times=[.499,.5,.501,0.,1.,.5]))
    put('pending_readout','latency',record([segment(duration=2.2,readout=[branch(1j*X)])],pending))
    put('one_photon','latency',record([segment(duration=2.4,emission=[enc(E)],eta=1.)],excited))
    put('unobserved_emission','conditioning',record([segment(duration=2.,emission=[enc(E)],eta=.23)],excited))
    put('blocked_electronics','latency',record([segment(duration=2.,emission=[enc(E)],bandwidth=0.)],excited))
    put('fixed_renewal','cumulants',record([segment(duration=3.,dark=.8,bandwidth=1.7,readout=[branch(np.eye(2),.65)])]))
    put('first_readout_stopped','conditioning',record([segment(duration=.7,dark=.8,bandwidth=2.5,readout=[branch(expm(-.4j*X),.6)]),
        segment(duration=1.2,H=enc([h,.3*X,-h]),bandwidth=0.)],times=[.2,.7,1.,1.3,1.9]))
    put('stretched_recovery','clock',record([segment(duration=2.8,bandwidth=1.4,readout=[branch(np.eye(2),.35,.35)])],pending,
        times=[0.,.349,.35,.8,1.6,2.8]))
    put('compressed_recovery','clock',record([segment(duration=1.4,bandwidth=1.9,readout=[branch(np.eye(2),1.3,-.65)])],pending,
        times=[0.,1.,1.299,1.3,1.35,1.4]))
    put('projective_branch_memory','instrument',record([segment(duration=2.3,bandwidth=1.5,
        readout=[branch(P0,.35,.2),branch(P1,1.2,-.3)])],pending))
    put('weak_branch_feedback','instrument',record([segment(duration=2.4,H=enc([h,-.3*X,.6*Z]),
        emission=[enc(.7*E)],bath=[enc(.4*E.T)],dark=.3,bandwidth=2.1,
        readout=[branch(weak[0],.4,.2),branch(weak[1],1.3,-.3)])],pending))
    put('later_entries_return_first','clock',record([segment(duration=.8,bandwidth=2.2,readout=[branch(np.eye(2),1.35,-.1)]),
        segment(duration=1.6,bandwidth=2.2,readout=[branch(np.eye(2),.35,.2)])],pending,
        times=[0.,.8,1.15,1.35,1.7,2.07,2.4]))
    put('overlapping_mark_windows','clock',record([segment(duration=.7,dark=.5,bandwidth=3.,readout=[branch(P0,.4,.5),branch(P1,1.3,-.6)]),
        segment(duration=.6,H=enc([h,X,-h]),bandwidth=2.,readout=[branch(weak[0],.45,.8),branch(weak[1],.8,-.5)]),
        segment(duration=1.,dark=.8,bandwidth=1.2,readout=[branch(P0,1.1,-.5),branch(P1,.35,.7)])],pending))
    put('dead_quantum_transport','coherence',record([segment(duration=.55,H=enc([.2*X,h,-h]),bandwidth=2.7,
        readout=[branch(weak[0],.4,1.),branch(weak[1],1.4,-.6)]),
        segment(duration=.63,H=enc([h,-.5*X,1.2*X]),emission=[enc(.7*E.T)],bath=[enc(.2*Z)],eta=.2,bandwidth=.4),
        segment(duration=1.12,H=enc([.3*Z,.7*X,h]),emission=[enc(.8*E)],dark=.8,bandwidth=2.2,
        readout=[branch(expm(-.4j*X),.4,.5)])],pending,times=[0.,.55,.9,1.18,1.4,1.5,2.03,2.3]))
    put('dead_emission_tally','cumulants',record([segment(duration=2.7,emission=[enc(.9*E)],bath=[enc(.7*E.T)],eta=.85,dark=.3,bandwidth=3.2,
        readout=[branch(P0,.4,.3),branch(P1,1.4,-.35)])],dead,remaining=.45))
    put('initial_and_new_cohorts','memory',record([segment(duration=1.8,H=enc([h,-h,.4*X]),emission=[enc(.7*E)],bath=[enc(.3*Z)],dark=.4,bandwidth=2.3,
        readout=[branch(weak[0],.35,.6),branch(weak[1],1.3,-.4)])],
        np.array([np.diag([.15,.05]),coherent*.45,np.diag([.1,.25])]),remaining=0.,times=[0.,.2,.35,.7,1.3,1.8]))
    put('rare_first_readout','conditioning',record([segment(duration=2.8,H=enc([h,h,-h]),emission=[enc(E)],eta=1e-4,dark=1e-5,bandwidth=.002,
        readout=[branch(weak[0],.4,.3),branch(weak[1],1.3,-.3)])],excited))
    rng=np.random.default_rng(20261024)
    def matrix(d):return rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
    def random_segment(d,duration,maximum=False):
        hs=[]
        for q in range(3):
            z=matrix(d);z=(z+z.conj().T)/2;hs.append(z/np.linalg.norm(z,2)*(3 if maximum else rng.uniform(.3,2.)))
        ops=[matrix(d) for _ in range(2 if maximum else 1)]
        baths=[matrix(d) for _ in range(2 if maximum else 1)]
        scale=np.sqrt((3 if maximum else rng.uniform(.5,2.4))/sum(np.linalg.norm(z,2)**2 for z in ops+baths))
        raw=[matrix(d) for _ in range(3)];S=sum(K.conj().T@K for K in raw)
        values,vectors=np.linalg.eigh(S);inv=(vectors/np.sqrt(values))@vectors.conj().T
        marks=[]
        for K in raw:
            base=float(rng.uniform(.35,1.45));end=float(rng.uniform(.35,1.45))
            slope=float(np.clip((end-base)/duration,-.65,1.3))
            marks.append(branch(K@inv,base,slope))
        return segment(d,duration,H=enc(hs),emission=[enc(scale*z) for z in ops],bath=[enc(scale*z) for z in baths],
            eta=.9 if maximum else float(rng.uniform(.1,.9)),dark=1. if maximum else .4,
            bandwidth=6. if maximum else float(rng.uniform(.7,4.5)),readout=marks)
    for k in range(5):
        d=2+k%2;blocks=[]
        for q in range(3):z=matrix(d);blocks.append(z@z.conj().T)
        rho=np.array(blocks);rho/=sum(z.trace().real for z in rho)
        segments=[random_segment(d,float(rng.uniform(.35,.9))) for j in range(2+k%3)]
        T=sum(s['duration'] for s in segments);remaining=float(rng.uniform(.05,1.1))
        times=[T,0.,remaining,T*.37,T*.8]+np.cumsum([s['duration'] for s in segments]).tolist()
        put('random_instrument_%02d'%k,'mixed',record(segments,rho,remaining,times))
    max_segments=[random_segment(3,1.5,True) for _ in range(4)]
    rho=np.zeros((3,3,3),complex);rho[0]=np.eye(3)*.1;rho[1]=np.eye(3)*.15;rho[2]=np.eye(3)/12
    put('maximum_domain','mixed',record(max_segments,rho,remaining=.2,times=[6.,0.,.2,1.5,1.8,3.,3.3,4.5,4.8,6.]))
    return rows

def cases():return json.loads((Path(__file__).parent/'fixtures.json').read_text())
