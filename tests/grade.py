"""Privileged independent verifier; candidate CLI always runs as uid 1000."""
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cases import cases
from checker import expected,verify

def read_output(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'r',encoding='utf-8') as f:
        info=os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>4_000_000: raise ValueError('invalid output file')
        return json.load(f)

def stop_solver_processes():
    # Agent daemons must not survive into hidden verification or subsequent cases.
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            if p.stat().st_uid==1000: os.kill(int(p.name),signal.SIGKILL)
        except (ProcessLookupError,FileNotFoundError,PermissionError): pass

def main():
    stop_solver_processes()
    records=cases(); assert len(records)<=28
    wants=[expected(data) for _,_,data in records]
    rows=[]; consumed=0.
    for (name,group,data),want in zip(records,wants):
        failures=[]; remaining=240.-consumed
        if remaining<=0:
            rows.append({'id':name,'group':group,'passed':False,'failures':['240 second batch budget exhausted']}); continue
        with tempfile.TemporaryDirectory(prefix='photons-') as directory:
            work=Path(directory); os.chown(work,1000,1000); os.chmod(work,0o700)
            source,dest=work/'input.json',work/'output.json'
            source.write_text(json.dumps(data),encoding='utf-8'); os.chmod(source,0o644)
            start=time.monotonic(); elapsed=None
            proc=subprocess.Popen([sys.executable,'/app/photon_report.py',str(source),str(dest)],
                cwd='/app',stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                user=1000,group=1000,extra_groups=[],start_new_session=True,
                env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/home/solver','LANG':'C.UTF-8',
                     'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
            try:
                code=proc.wait(timeout=remaining); elapsed=time.monotonic()-start
                if code: failures.append('CLI exited %s'%code)
                else: failures.extend(verify(data,read_output(dest),want))
            except subprocess.TimeoutExpired: failures.append('240 second batch budget exhausted')
            except (OSError,ValueError,TypeError,OverflowError,RecursionError): failures.append('Invalid output')
            finally:
                stop_solver_processes(); proc.wait()
                consumed+=elapsed if elapsed is not None else time.monotonic()-start
        rows.append({'id':name,'group':group,'passed':not failures,'failures':failures})
    result={'passed':sum(r['passed'] for r in rows),'total':len(rows),'candidate_seconds':consumed,'cases':rows}
    Path('/logs/verifier/details.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if all(r['passed'] for r in rows) else 1

if __name__=='__main__': raise SystemExit(main())
