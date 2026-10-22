"""Run a TRUSTED pytest suite with fresh, complete execution evidence.

This is a completion gate, not a sandbox. Tests must invoke candidate code in a
separate restricted process. Never import candidate code into this worker.
freeze executes trusted collection; run executes trusted tests. Neither belongs
in a model's writable environment. pytest is required in this interpreter.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile


def no_links(path):
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError("linked paths are not allowed")
    return path


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    path = no_links(path)
    if not path.is_file() or path.stat().st_size > 8_000_000:
        raise ValueError("missing or oversized JSON evidence")
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs)


def atomic_write(path, value):
    path = no_links(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def suite_digest(suite, expected):
    digest = hashlib.sha256()
    count = 0
    for path in sorted(suite.rglob("*"), key=lambda p: p.relative_to(suite).as_posix()):
        no_links(path)
        if path.suffix in (".pyc", ".pyo") or any(p in ("__pycache__", ".pytest_cache") for p in path.relative_to(suite).parts):
            # -B prevents writing bytecode, not loading old timestamp caches.
            # Never silently omit executable bytes from the frozen suite.
            raise ValueError("trusted suite contains cache artifacts")
        if path == expected:
            continue
        if not path.is_file():
            continue
        count += 1
        name = path.relative_to(suite).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    if count == 0:
        raise ValueError("empty trusted suite")
    return digest.hexdigest()


def nodeids(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(v, str) and bool(v.strip()) for v in value)
            and len(set(value)) == len(value))


def validate_completion(report, expected, run_id, digest, returncode):
    """Validate only structural completion; trust must come from isolation."""
    reasons = []
    if not isinstance(expected, dict) or expected.get("schema_version") != 1 or not nodeids(expected.get("nodeids")):
        return ["invalid_expected_manifest"]
    if expected.get("suite_sha256") != digest:
        reasons.append("suite_not_frozen_version")
    if returncode != 0:
        reasons.append("worker_nonzero_or_timeout")
    if not isinstance(report, dict):
        return reasons + ["missing_completion_report"]
    if report.get("schema_version") != 1 or report.get("run_id") != run_id or report.get("suite_sha256") != digest:
        reasons.append("report_identity_mismatch")
    if report.get("finished") is not True or type(report.get("exitstatus")) is not int or report["exitstatus"] != 0:
        reasons.append("session_not_successfully_finished")
    if report.get("collection_errors") != []:
        reasons.append("collection_errors_or_missing_evidence")
    collected = report.get("collected")
    if not nodeids(collected) or set(collected) != set(expected["nodeids"]):
        reasons.append("collected_set_mismatch")
    records = report.get("outcomes")
    seen = set()
    if not isinstance(records, list):
        return reasons + ["missing_phase_reports"]
    for record in records:
        if not isinstance(record, dict):
            reasons.append("invalid_phase_report")
            continue
        key = (record.get("nodeid"), record.get("when"))
        if not all(isinstance(v, str) for v in key):
            reasons.append("invalid_phase_identity")
            continue
        if key in seen:
            reasons.append("duplicate_phase_report")
        seen.add(key)
        if record.get("outcome") != "passed" or record.get("wasxfail") is not False:
            reasons.append("failed_skipped_or_xfail_phase")
    required = {(case, stage) for case in expected["nodeids"] for stage in ("setup", "call", "teardown")}
    if seen != required:
        reasons.append("executed_phase_set_mismatch")
    return sorted(set(reasons))


class CompletionPlugin:
    def __init__(self, run_id, digest):
        self.data = dict(schema_version=1, run_id=run_id, suite_sha256=digest,
                         collected=[], outcomes=[], collection_errors=[], finished=False, exitstatus=None)

    def pytest_collection_finish(self, session):
        self.data["collected"] = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.data["collection_errors"].append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        self.data["outcomes"].append(dict(nodeid=report.nodeid, when=report.when,
                                          outcome=report.outcome, wasxfail=hasattr(report, "wasxfail")))

    def pytest_sessionfinish(self, session, exitstatus):
        self.data["finished"] = True
        self.data["exitstatus"] = int(exitstatus)


def worker(suite, directory, run_id, digest, collect):
    import pytest
    plugin = CompletionPlugin(run_id, digest)
    arguments = [str(suite), "--rootdir", str(suite), "-c", str(directory / "pytest.ini"),
                 "--noconftest", "--import-mode=importlib", "-p", "no:cacheprovider", "-q"]
    if collect:
        arguments.append("--collect-only")
    code = int(pytest.main(arguments, plugins=[plugin]))
    atomic_write(directory / "completion.json", json.dumps(plugin.data, allow_nan=False) + "\n")
    return code


def launch(suite, directory, run_id, digest, timeout, collect=False):
    # Credentials and user-controlled Python/pytest settings are not inherited.
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "LANG", "LC_ALL", "TZ"}}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    (directory / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "_worker",
               str(suite), str(directory), run_id, digest, "collect" if collect else "run"]
    with (directory / "pytest-output.txt").open("wb") as log:
        proc = subprocess.Popen(command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        def stop():
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif proc.poll() is None:
                proc.kill()  # Windows smoke tests only; no descendant-isolation claim.
            proc.wait()
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop()
            return None
        except BaseException:
            stop()
            raise


def _run_gate(suite, expected_path, output, timeout, collect=False):
    # Establish a safe output location first, then clear the score even if the
    # suite, manifest or timeout subsequently turns out to be invalid.
    output = no_links(output)
    suite, expected_path = (Path(os.path.abspath(p)) for p in (suite, expected_path))
    skill = Path(__file__).resolve().parent
    if output.is_relative_to(suite) or output.is_relative_to(skill) or expected_path.is_relative_to(output):
        raise ValueError("run output must be outside suite, manifest and skill")
    if collect and expected_path.exists():
        raise ValueError("freeze requires a new manifest outside the skill")
    output.mkdir(parents=True, exist_ok=True)
    reward = output / "reward.txt"
    no_links(reward)
    if not collect:
        atomic_write(reward, "0\n")  # In this explicitly selected run output, never reuse an old score.
    run_id = secrets.token_hex(24)
    directory = output / ("completion-" + run_id)
    directory.mkdir(mode=0o700)
    reasons, code, report = [], None, None
    digest = None
    expected = None
    try:
        suite, expected_path = no_links(suite), no_links(expected_path)
        if not suite.is_dir() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("existing trusted suite and positive finite timeout required")
        digest = suite_digest(suite, expected_path)
        if not collect:
            expected = read_json(expected_path)
            if not isinstance(expected, dict) or expected.get("suite_sha256") != digest or not nodeids(expected.get("nodeids")):
                raise ValueError("expected manifest does not match current suite")
        code = launch(suite, directory, run_id, digest, timeout, collect)
        if (directory / "completion.json").exists():
            report = read_json(directory / "completion.json")
        if suite_digest(suite, expected_path) != digest:
            reasons.append("suite_changed_during_execution")
        if collect:
            if (code != 0 or not isinstance(report, dict) or report.get("finished") is not True
                    or report.get("exitstatus") != 0 or report.get("collection_errors") != []
                    or not nodeids(report.get("collected")) or report.get("run_id") != run_id):
                reasons.append("collection_not_complete")
            if not reasons:
                expected = dict(schema_version=1, suite_sha256=digest, nodeids=report["collected"])
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation prevents accidental replacement of a frozen list.
                with expected_path.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(expected, stream, indent=2)
                    stream.write("\n")
        else:
            reasons += validate_completion(report, expected, run_id, digest, code)
    except (OSError, ValueError, KeyboardInterrupt):
        reasons.append("setup_or_evidence_error")
    result = dict(schema_version=1, operation="freeze" if collect else "run", run_id=run_id,
                  suite_sha256=digest, worker_returncode=code, accepted=not reasons,
                  reasons=sorted(set(reasons)), evidence_directory=str(directory),
                  scope="trusted pytest completion only; sandbox and scientific correctness not certified")
    atomic_write(directory / "gate-result.json", json.dumps(result, indent=2) + "\n")
    if not collect and not reasons:
        atomic_write(reward, "1\n")
    return result


def run_gate(suite, expected_path, output, timeout, collect=False):
    output = no_links(output)
    suite_absolute = Path(os.path.abspath(suite))
    manifest_absolute = Path(os.path.abspath(expected_path))
    if (output.is_relative_to(suite_absolute)
            or output.is_relative_to(Path(__file__).resolve().parent)
            or manifest_absolute.is_relative_to(output)):
        raise ValueError("output must be isolated from input and skill")
    output.mkdir(parents=True, exist_ok=True)
    lock = no_links(output / ".completion.lock")
    # A directory is owned by one trial. Refuse concurrent writers, never
    # clear another active trial's score or delete its lock on contention.
    try:
        with lock.open("x", encoding="ascii") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise ValueError("output is in use or has a stale lock; use a new trial directory") from None
    try:
        return _run_gate(suite, expected_path, output, timeout, collect)
    finally:
        lock.unlink()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "_worker":
        _, suite, directory, run_id, digest, mode = args
        return worker(Path(suite), Path(directory), run_id, digest, mode == "collect")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "run"))
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=270)
    parsed = parser.parse_args(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupted)
    try:
        result = run_gate(parsed.suite, parsed.expected, parsed.out_dir, parsed.timeout, parsed.operation == "freeze")
    except (OSError, ValueError):
        print(json.dumps(dict(accepted=False, reasons=["invalid_paths_or_configuration"])))
        return 2
    print(json.dumps(result))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
