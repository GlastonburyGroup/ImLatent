#!/bin/bash
#SBATCH --job-name=gpurun
#SBATCH --partition=gpuq        # adapt to your cluster
#SBATCH --time=30-00:00:0       # walltime
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9       # 9 CPUs -> num_workers resolves to 8 (MainEngine caps it at cpus_avail-1), as in the original runs
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=15000Mb
#SBATCH --output=gpurun_%x_%j.log

# Generic SLURM GPU wrapper: runs a python script of this repository inside its uv environment.
# usage: sbatch gpurun_uv.sh --root <repo root> --programme <script.py, relative to root> --args "<arguments for the script>"

exec 2>&1
echo "Command executed: $0 $@"

function usage {
    echo "usage: $0 --root string --programme string --args \"string\""
    echo "  --root       Repository root (the folder containing pyproject.toml)"
    echo "  --programme  Python file to run, relative to the root (e.g. Executors/main_diffAE2D.py)"
    echo "  --args       Arguments for the programme, as a single quoted string"
}

while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        usage; exit 0
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

root="${root:-}"
programme="${programme:-}"
args="${args:-}"

[ -z "$root" ] && { echo "Error: --root is required"; exit 1; }
[ -z "$programme" ] && { echo "Error: --programme is required"; exit 1; }

cd "$root" || exit 1
echo "Starting the execution of $programme with uv (root: $root)"
srun uv run python $programme $args
