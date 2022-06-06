import os
import re

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DAIC_DIR = os.path.join(os.getcwd(), 'data', 'DAIC-WoZ')

with open(os.path.join(DAIC_DIR, 'Info', 'queries.txt')) as f:
    queries = f.readlines()
    

def identify_topics(sentence):
    for query in queries:
        query = query.strip('\n')
        sentence = sentence.strip('\n')
        if query == sentence:
            return True
    return False


def preprocess(participant_id):
    transcript = pd.read_csv(os.path.join(DAIC_DIR, 'DAIC', f'{participant_id}_P/{participant_id}_TRANSCRIPT.csv'), sep='\t').fillna('')
    audio, sr = librosa.load(os.path.join(DAIC_DIR, 'DAIC', f'{participant_id}_P/{participant_id}_AUDIO.wav'), sr=None)
    
    response = ''
    signal = []
    signals = []
    responses = []
    for trans in transcript.itertuples():
        if getattr(trans, 'speaker') == 'Ellie':
            if '(' in getattr(trans, 'value'):
                content = re.findall(re.compile(r'[(](.*?)[)]', re.S), getattr(trans,'value'))[0]
            else:
                content = getattr(trans, 'value').strip()
                
            if identify_topics(content) or 'asked everything i need to' in getattr(trans, 'value'):
                if len(response) != 0 and len(signal) != 0:
                    responses.append(response.strip())
                    signals.append(signal)
                    signal = []
                    response = ''
        elif getattr(trans, 'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(trans, 'value'):
                continue
            response += ' ' + re.sub(r'<[a-zA-Z\ ]*>', '', getattr(trans, 'value').split('\n')[0]).strip()
            response = response.replace('  ', '')
            response.strip()
            
            start_time = int(getattr(trans, 'start_time') * sr)
            stop_time = int(getattr(trans, 'stop_time') * sr)
            signal = np.hstack((signal, audio[start_time: stop_time].astype(np.float32)))
            
    responses = np.array(responses, dtype=object)
    signals = np.array(signals, dtype=object)
    return responses, signals


if __name__ == '__main__':
    
    info_dict = {
        'train': os.path.join(DAIC_DIR, 'Info', 'train_split_Depression_AVEC2017.csv'), 
        'val': os.path.join(DAIC_DIR, 'Info', 'dev_split_Depression_AVEC2017.csv'),
        'test': os.path.join(DAIC_DIR, 'Info', 'full_test_split.csv')
    }
    
    for data_type in ['train', 'val', 'test']:
        infos = ["participant_id\tresponse_num\tcls\treg\n"]
        df = pd.read_csv(info_dict[data_type])
        participant = df[['Participant_ID']]['Participant_ID'].tolist()
        cls_label = df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()
        reg_label = df[['PHQ8_Score']]['PHQ8_Score'].tolist()
        
        whole_audio_feats, whole_text_feats, whole_targets = [], [], []
        for index in tqdm(range(len(participant))):
            participant_id = participant[index]
            if participant_id in [451, 458, 480, 318, 321, 341, 362]:
                continue
            cls = cls_label[index]
            reg = reg_label[index]
            responses, signals = preprocess(participant_id)
            out_path = os.path.join(DAIC_DIR, 'preprocessed', f'{data_type}', f"{participant_id}_P.npz")
            np.savez(
                out_path,
                responses=responses,
                signals=signals,
                cls=cls,
                reg=reg
            )
            infos.append(f"{participant_id}\t{len(responses)}\t{cls}\t{reg}\n")
        with open(os.path.join(DAIC_DIR, 'preprocessed', f'{data_type}.csv'), mode='w') as f:
            f.writelines(infos)
