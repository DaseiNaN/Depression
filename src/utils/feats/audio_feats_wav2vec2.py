import argparse
import collections
import glob
import itertools
import os
import sys

import librosa
import numpy as np
import tensorflow as tf
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append(os.getcwd())

import src.vendor.loupe_keras as lpk  # noqa: E402
from transformers import AutoModelForPreTraining

zh_wav2vec2_model = AutoModelForPreTraining.from_pretrained("/home/dasein/Projects/Depression/resources/TencentGameMate/chinese-wav2vec2-large")


def main(data_dir: str, out_dir: str, ssds_threshold: int, cluster_size: int, augment: bool):
    """Extract feats for audio.

    Args:
        data_dir (str): Directory of EATD-Corpus
        out_dir (str): Directory to save .npz file
        ssds_threshold (int): Threshold for standard SDS score
                              正常     :       ssds < 53
                              轻度     : 53 <= ssds < 59
                              中度至重度: 60 <= ssds < 69
                              重度     : 70 <= ssds
        cluster_size (int): Cluster size for NetVLAD

    Returns:
        None
    """
    feats = []
    targets = []

    for item_dir in sorted(
        glob.glob(data_dir + r"/*/*"), reverse=False, key=lambda x: int(x.split("_")[-1])
    ):
        if "Feats" in item_dir:
            continue
        
        # 1. Load standard SDS score as target
        with open(item_dir + r"/new_label.txt") as file:
            target = float(file.readline())
            
        feat_dict = collections.defaultdict(lambda: None)
        
        # 2. Extract 256-dim feat
        for item_path in glob.glob(item_dir + r"/*_out.wav"):
            y, sr = librosa.load(item_path, sr=None)
            dur = len(y) / sr
            if len(y) == 0:
                y = np.array([1e-4] * sr * 5)
            y = torch.from_numpy(y).unsqueeze(dim=0)
            y = y.to(torch.float32)
            y_vec = zh_wav2vec2_model(y).projected_states
            y_vec = y_vec.squeeze(dim=0)
            y_vec = y_vec.detach().numpy()
            
            # wav2vec2 vec -> vlad feat
            max_samples, feature_size = y_vec.shape
            vlad_feat = lpk.NetVLAD(
                feature_size=feature_size,
                max_samples=max_samples,
                cluster_size=cluster_size,
                output_dim=cluster_size * 16,
            )(tf.convert_to_tensor(y_vec)).numpy()

            vlad_feat = np.squeeze(vlad_feat, axis=0)
            item_type = item_path.split("/")[-1].replace("_out.wav", "")
            feat_dict[item_type] = vlad_feat

        if augment is True and target >= ssds_threshold:
            targets.extend([target] * 6)
            for k_perm in itertools.permutations(["positive", "neutral", "negative"]):
                feats.append([feat_dict[k] for k in list(k_perm)])
        else:
            targets.append(target)
            feats.append([feat_dict[k] for k in ["positive", "neutral", "negative"]])
        
    feats = np.array(feats)
    targets = np.array(targets)
    dep_idxs = np.where(targets >= ssds_threshold)[0]
    non_idxs = np.where(targets < ssds_threshold)[0]
    print(
        "Number of samples: {} ( dep-{}, non dep-{} )".format(
            feats.shape[0], len(dep_idxs), len(non_idxs)
        )
    )
    np.savez(
        os.path.join(out_dir, "audio_feats.npz"),
        feats=feats,
        targets=targets,
        dep_idxs=dep_idxs,
        non_idxs=non_idxs,
    )
    print("audio_feats.npz has been saved in dir: {}".format(out_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--data_dir", type=str, help="Directory of EATD-Corpus", required=True
    )
    parser.add_argument(
        "-o", "--out_dir", type=str, help="Directory to save audio_feats.npz file", default=None
    )
    parser.add_argument(
        "-t", "--ssds_threshold", type=int, help="Threshold for standard SDS score", default=53
    )
    parser.add_argument(
        "-c", "--cluster_size", type=int, help="Cluster size for NetVLAD", default=16
    )
    parser.add_argument(
        "-a", "--augment", type=bool, help="Data Augment, default=False", default=False
    )

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(os.getcwd(), "data/EATD-Feats")
    opts = vars(args)
    main(**opts)
