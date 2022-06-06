#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/daic.sh

stage=1

if [ $stage -eq 0 ]; then
    echo 'Extracting DAIC-WoZ feats (audio & text)'
    python src/utils/feats/daic_preprocess.py
    python src/utils/feats/daic_feats.py
fi

if [ $stage -eq 1 ]; then
    python examples/DAIC-WoZ/cls_train.py
fi

