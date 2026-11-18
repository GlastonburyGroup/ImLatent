#!/bin/bash
# Thin wrapper: see submit_chain_abdominal.sh for all options.
exec "$(dirname "${BASH_SOURCE[0]}")/submit_chain_abdominal.sh" --dataset 20254 "$@"
