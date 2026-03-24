#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <directory> [mask.py options]"
  echo "Example: $0 ./images --text statue --invert --out-dir ./masked"
  exit 1
fi

DIR="$1"
shift

if [[ ! -d "$DIR" ]]; then
  echo "Error: directory not found: $DIR"
  exit 1
fi

python mask.py --dir "$DIR" "$@"
