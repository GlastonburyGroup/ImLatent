#!/bin/bash
# Submits every run of the final table: 12 dataset/organ groups (incl. F20204 and F20254 full image) x 5 seeds x 3 latent sizes = 180 chains.
# F20259 full image is not part of the final table (no finished run); submit it manually if needed.
# usage: run_all_abdominal.sh [--dry-run] [any extra option for submit_chain_abdominal.sh, e.g. --extra_args "--wnbentity me"]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=false
if [ "$1" == "--dry-run" ]; then DRY=true; shift; fi

SEEDS="1701 1993 1994 2023 42"
DIMS="128 64 32"
RUNS=(
  "20204 full Liver" "20204 crop Liver" "20204 crop Spleen" "20204 crop Aorta"
  "20254 full Liver" "20254 crop Liver" "20254 crop Spleen" "20254 crop Aorta"
  "20259 crop Pancreas" "20259 crop Spleen" "20259 crop Aorta" "20259 crop Kidney"
)
for r in "${RUNS[@]}"; do
  read -r ds dtype organ <<< "$r"
  for seed in $SEEDS; do
    for dim in $DIMS; do
      cmd=("$SCRIPT_DIR/submit_chain_abdominal.sh" --dataset "$ds" --data_type "$dtype" --wnbgroup "$organ" --seed "$seed" --latent_dim "$dim" "$@")
      if $DRY; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
    done
  done
done
