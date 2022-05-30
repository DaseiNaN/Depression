#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

stage=4

data_dir=$PWD/data/EATD-Corpus
origin_out_dir=$PWD/data/EATD-Feats/origin
augment_out_dir=$PWD/data/EATD-Feats/augment

if [ $stage -eq 0 ]; then
    echo 'Extracting EATD feats (audio & text)'
    # Extract audio feats
    python src/utils/feats/audio_feats.py -i $data_dir -o $origin_out_dir -a False
    python src/utils/feats/audio_feats.py -i $data_dir -o $augment_out_dir -a True
    # Extract text feats
    python src/utils/feats/text_feats.py -i $data_dir -o $origin_out_dir -a False
    python src/utils/feats/text_feats.py -i $data_dir -o $augment_out_dir -a True
fi

if [ $stage -eq 1 ]; then
    python examples/cls_train.py
fi

if [ $stage -eq 2 ]; then
    python examples/reg_train.py
fi

if [ $stage -eq 3 ]; then
    python examples/fused_cls_train.py
fi

if [ $stage -eq 4 ]; then
    python examples/fused_reg_train.py
fi