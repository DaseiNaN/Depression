
import logging
import os
import re
import sys
from cgitb import text

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

# from elmoformanylangs import Embedder

sys.path.append(os.getcwd())
import src.vendor.loupe_keras as lpk  # noqa: E402

logging.basicConfig(level=logging.ERROR)

DAIC_DIR = os.path.join(os.getcwd(), 'data', 'DAIC-WoZ')
with open(os.path.join(DAIC_DIR, 'Info', 'queries.txt')) as f:
    queries = f.readlines()
    
# elmo = Embedder(os.path.join(os.getcwd(), "resources/zhs.model"), batch_size=2)

from transformers import AutoModelForPreTraining

zh_wav2vec2_model = AutoModelForPreTraining.from_pretrained("/home/dasein/Projects/Depression/resources/TencentGameMate/chinese-wav2vec2-large")


def identify_topics(sentence):
    for query in queries:
        query = query.strip('\n')
        sentence = sentence.strip('\n')
        if query == sentence:
            return True
    return False

def wav2vald(y, sr=16000, cluster_size=16):
    mel_spec = librosa.feature.melspectrogram(y=y, n_mels=80, sr=sr).T
    log_mel_spec = np.log(np.maximum(1e-6, mel_spec))
    max_samples, feature_size = log_mel_spec.shape
    vlad_feat = lpk.NetVLAD(
        feature_size=feature_size,
        max_samples=max_samples,
        cluster_size=cluster_size,
        output_dim=cluster_size * 16,
    )(tf.convert_to_tensor(log_mel_spec)).numpy()
    
    vlad_feat = np.squeeze(vlad_feat, axis=0)
    return vlad_feat

def wav2vec2_feats(y, sr=16000, cluster_size=16):
    y_max_length = sr * 120
    if len(y) > y_max_length:
        y = y[:y_max_length]
    y = torch.from_numpy(y).unsqueeze(dim=0)
    y = y.to(torch.float32)
    y_vec = zh_wav2vec2_model(y).projected_states
    y_vec = y_vec.squeeze(dim=0)
    y_vec = y_vec.detach().numpy()
    max_samples, feature_size = y_vec.shape
    vlad_feat = lpk.NetVLAD(
        feature_size=feature_size,
        max_samples=max_samples,
        cluster_size=cluster_size,
        output_dim=cluster_size * 16,
    )(tf.convert_to_tensor(y_vec)).numpy()
    vlad_feat = np.squeeze(vlad_feat, axis=0)
    return vlad_feat
    
def extract_feats(participant_id, data_type, whole_audio_feats=[], whole_text_feats=[], whole_cls_label=[], whole_reg_label=[]):
    data = np.load(os.path.join(DAIC_DIR, 'preprocessed', data_type, f'{participant_id}_P.npz'), allow_pickle=True)
    responses = data["responses"]
    signals = data["signals"]
    cls = data["cls"]
    reg = data["reg"]
    
    if data_type == "train" and cls == 1 and reg >= 10:
        times = len(responses) // 10
        for i in range(times):
            seg_signal = signals[i*10:(i+1)*10]
            audio_feat = [wav2vec2_feats(np.array(item)) for item in seg_signal]
            whole_audio_feats.append(audio_feat)
            
            # seg_response = responses[i*10:(i+1)*10]
            # text_feat = [np.array(item).mean(axis=0) for item in elmo.sents2elmo(seg_response)]
            # whole_text_feats.append(text_feat)
            
            whole_cls_label.append(cls)
            whole_reg_label.append(reg)
    else:
        seg_signal = signals[0:10]
        audio_feat = [wav2vec2_feats(np.array(item)) for item in seg_signal]
        whole_audio_feats.append(audio_feat)
        
        # seg_response = responses[0: 10]
        # text_feat = [np.array(item).mean(axis=0) for item in elmo.sents2elmo(seg_response)]
        # whole_text_feats.append(text_feat)
        
        whole_cls_label.append(cls)
        whole_reg_label.append(reg)

    print('{}_P feature done'.format(participant_id))
    

if __name__ == '__main__':
    csv_dict = {
        'train': os.path.join(DAIC_DIR, 'preprocessed', 'train.csv'), 
        'val': os.path.join(DAIC_DIR, 'preprocessed', 'val.csv'),
        'test': os.path.join(DAIC_DIR, 'preprocessed', 'test.csv'),
        'tmp': os.path.join(DAIC_DIR, 'preprocessed', 'tmp.csv')
    }
    
    for data_type in ['test', 'val', 'train']:
        df = pd.read_csv(csv_dict[data_type], sep='\t')
        participant = df[['participant_id']]['participant_id'].tolist()
        whole_audio_feats, whole_text_feats, whole_cls_label, whole_reg_label = [], [], [], []
        
        for index in range(len(participant)):
            participant_id = participant[index]
            extract_feats(participant_id,  data_type, whole_audio_feats, whole_text_feats, whole_cls_label, whole_reg_label)
            
        count = pd.value_counts(whole_cls_label)
        
        whole_audio_feats = np.array(whole_audio_feats)
        # whole_text_feats = np.array(whole_text_feats)
        whole_cls_label = np.array(whole_cls_label)
        whole_reg_label = np.array(whole_reg_label)
        
        out_path = os.path.join(os.getcwd(), 'data', 'DAIC-WoZ-Feats', 'audio', 'wav2vec2', f"{data_type}_audio_feats.npz")
        
        np.savez(
            out_path,
            feats=whole_audio_feats,
            cls_label=whole_cls_label,
            reg_label=whole_reg_label
        )
        print("{}_feats.npz has been saved in: {}".format(data_type, out_path))
        print(whole_audio_feats.shape, 'dep={}, non_dep={}'.format(count[1], count[0]))
