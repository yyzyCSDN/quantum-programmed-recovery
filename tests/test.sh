#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
chown root:root /logs/verifier /tests
chmod 0755 /logs/verifier
chmod 0700 /tests
reward=/logs/verifier/reward.txt
write_reward() {
    local temporary
    temporary=$(mktemp /logs/verifier/.reward.XXXXXX)
    printf '%s\n' "$1" > "$temporary"
    chmod 0644 "$temporary"
    mv -fT "$temporary" "$reward"
}
write_reward 0
completed=0
finish() { if [ "$completed" -ne 1 ]; then write_reward 0; fi; }
trap finish EXIT
trap 'exit 1' INT TERM HUP
if timeout --signal=TERM --kill-after=5s 280s python -I /tests/grade.py; then
    write_reward 1
fi
completed=1
