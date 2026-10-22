#!/bin/bash
set -euo pipefail
umask 022
reward=/logs/verifier/reward.txt
# Invalidate stale success before any ownership, suite, or interpreter setup.
mkdir -p /logs/verifier
printf '0\n' > "$reward"
finish() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf '0\n' > "$reward" || printf '%s\n' 'INFRASTRUCTURE_ERROR: reward directory is not writable' >&2
    fi
}
trap finish EXIT
trap 'exit 1' INT TERM HUP
if [ "$(id -u)" -ne 0 ]; then
    printf '%s\n' 'INFRASTRUCTURE_ERROR: verifier requires root; candidate runs as solver' >&2
    exit 1
fi
chown root:root /logs/verifier /tests
chmod 0755 /logs/verifier
chmod 0700 /tests
# Only the trusted gate may grant full reward after complete pytest evidence.
python -I -B /tests/completion.py run --suite /tests --expected /tests/expected-tests.json --out-dir /logs/verifier --timeout 275
