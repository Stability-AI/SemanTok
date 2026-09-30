#!/usr/bin/env bash
# Fail if any tracked file mentions internal infrastructure. Run before every push (and in CI).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
pattern='/weka|/admin/|/fsx|home-mishok|mishok43|proj-mod3d|sai[-_]jumphost|sai_media|mod3d|p5en|sbatch|squeue|slurm|comet|stability\.ai|Stability-AI/ml-videoflextok|burst_buffer|tools/uco3d|tools/k600'
hits=$(git ls-files -z | grep -zv '^tools/leak_guard.sh$' | xargs -0 grep -nIiE "$pattern" || true)
if [[ -n "$hits" ]]; then
  echo "leak_guard: internal references found:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "leak_guard: clean"
