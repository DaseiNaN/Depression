#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

stage=3

data_dir=$PWD/data/EATD-Corpus
origin_out_dir=$PWD/data/EATD-Feats/origin
augment_out_dir=$PWD/data/EATD-Feats/augment
augment_wav2vec_out_dir=$PWD/data/EATD-Feats/augment_wav2vec2
origin_wav2vec_out_dir=$PWD/data/EATD-Feats/origin_wav2vec2

if [ $stage -eq 0 ]; then
    echo 'Extracting EATD feats (audio & text)'
    # Extract audio feats
    # python src/utils/feats/audio_feats.py -i $data_dir -o $origin_out_dir
    # python src/utils/feats/audio_feats.py -i $data_dir -o $augment_out_dir -a True
    # Extract text feats
    python src/utils/feats/text_feats.py -i $data_dir -o $origin_out_dir
    # python src/utils/feats/text_feats.py -i $data_dir -o $augment_out_dir -a True

    # Extract audio feats based on wav2vec2
    # python src/utils/feats/audio_feats_wav2vec2.py -i $data_dir -o $augment_wav2vec_out_dir -a True
    # python src/utils/feats/audio_feats_wav2vec2.py -i $data_dir -o $origin_wav2vec_out_dir
fi

if [ $stage -eq 1 ]; then
    # python examples/cls_train.py
    python examples/reg_train.py
fi

if [ $stage -eq 2 ]; then
    # python examples/fused_cls_train.py
    python examples/fused_reg_train.py
fi

if [ $stage -eq 3 ]; then
    python examples/attribute_cls_train.py
    # python examples/attribute_reg_train.py
fi
