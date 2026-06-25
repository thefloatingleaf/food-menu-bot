#!/bin/zsh
set -euo pipefail

cd "/Users/gg/Documents/Ayurveda/apps/household-inventory"
npm run add:usage-observation -- "$@"
