"""Trusted CLI fixture. Candidate source is never imported into this process."""
import ctypes
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time


def stop_candidates():
    """This verifier has its own PID namespace and exclusive candidate UID."""
    for item in Path('/proc').iterdir():
        if item.name.isdecimal():
            try:
                if item.stat().st_uid == 1000:
                    os.kill(int(item.name), signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError, PermissionError):
                pass


def reap_children():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                return
        except ChildProcessError:
            return


def read_output(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'r', encoding='utf-8') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('Output must be a bounded regular JSON file')
        return json.load(handle)


def run_cases(records, entry, limit, expected, verify):
    if os.geteuid() != 0:
        raise RuntimeError('Verifier must run as root; candidate runs as UID 1000')
    names = [name for name, _, _ in records]
    if not names or len(names) != len(set(names)):
        raise RuntimeError('Empty or duplicate case set')
    # Reap descendants that deliberately leave the original process group.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot establish child subreaper')
    stop_candidates()
    # All expectations are obtained from trusted mathematics before candidates run.
    for _, _, data in records:
        expected(data)
    rows = {}
    consumed = 0.0
    for name, group, data in records:
        row = {'id': name, 'group': group, 'returncode': None, 'errors': [], 'seconds': 0.0}
        rows[name] = row
        remaining = 240.0 - consumed
        if remaining <= 0:
            row['errors'].append('240 second batch budget exhausted')
            continue
        with tempfile.TemporaryDirectory(prefix='candidate-') as directory:
            work = Path(directory)
            os.chown(work, 1000, 1000)
            os.chmod(work, 0o700)
            source, output = work/'input.json', work/'output.json'
            source.write_text(json.dumps(data), encoding='utf-8')
            os.chmod(source, 0o644)
            began = time.monotonic()
            process = subprocess.Popen(
                [sys.executable, '-B', entry, str(source), str(output)], cwd='/app',
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                user=1000, group=1000, extra_groups=[], start_new_session=True,
                env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/home/solver',
                     'LANG': 'C.UTF-8', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            try:
                row['returncode'] = process.wait(timeout=remaining)
                row['seconds'] = time.monotonic() - began
                if row['returncode'] != 0:
                    row['errors'].append('CLI exited '+str(row['returncode']))
                else:
                    row['errors'].extend(verify(data, read_output(output, limit)))
            except subprocess.TimeoutExpired:
                row['errors'].append('240 second batch budget exhausted')
            except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
                row['errors'].append('Invalid output: '+type(exc).__name__)
            finally:
                stop_candidates()
                process.wait()
                reap_children()
                if not row['seconds']:
                    row['seconds'] = time.monotonic() - began
                consumed += row['seconds']
    report = {'total': len(rows), 'candidate_seconds': consumed, 'cases': list(rows.values())}
    Path('/logs/verifier/candidate-details.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return rows
