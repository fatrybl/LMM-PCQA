#!/bin/bash
# Builds the venv LMM-PCQA runs in: VAPE's environment through --system-site-packages (torch,
# Open3D, OpenCV and VAPE itself), the packages of venv_requirements.txt and the Q-Align fork
# checked out beside this one, installed without its dependencies. VAPE's environment does not
# change. pip honours PIP_CONSTRAINT, which pins the torch stack in VAPE's container.
#
#   usage: setup_venv.sh <python of VAPE's environment> <venv directory>
set -euo pipefail
fork=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
base_python=$1
venv=$2
"$base_python" -m venv --system-site-packages "$venv"
"$venv/bin/python" -m pip install --requirement "$fork/venv_requirements.txt"
"$venv/bin/python" -m pip install --no-deps --editable "$fork/../Q-Align"
"$venv/bin/python" -m pip check
