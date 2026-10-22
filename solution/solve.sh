#!/bin/bash
set -euo pipefail
install -m 0644 /solution/reference.py /app/photon_report.py
chown solver:solver /app/photon_report.py
