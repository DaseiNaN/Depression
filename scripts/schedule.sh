#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

stage=0

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
    python train.py experiment=classification
fi