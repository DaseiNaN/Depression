#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/daic.sh

stage=0

if [ $stage -eq 0 ]; then
    echo 'Extracting DAIC-WoZ feats (audio & text)'
    python src/utils/feats/daic_preprocess.py
    python src/utils/feats/daic_feats.py
fi

