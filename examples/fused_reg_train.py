import os
import sys

from isort import stream

sys.path.append(os.getcwd())
import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from src.datamodules.eatd_dataset import EATDDataset
from src.models.regression.audio_bilstm_net import AudioBiLSTMNet
from src.models.regression.fused_net import FusedNet
from src.models.regression.text_bilstm_net import TextBiLSTMNet
from src.utils.fused_loss import RegFusedLoss
from src.utils.metrics import measure_performance

AUDIO_NET_CONFIG = {
    "num_classes": 1,
    "dropout": 0.5,
    "num_layers": 2,
    "hidden_size": 256,
    "embed_size": 256
}

TEXT_NET_CONFIG = {
    "num_classes": 1,
    "dropout": 0.5,
    "num_layers": 2,
    "hidden_size": 128,
    "embed_size": 1024,
    "bidirectional": True
}

FUSED_NET_CONFIG = {
    "num_classes": 1,
    "text_hidden_dim": 128,
    "audio_hidden_dim": 256,
}

def set_logger(logger_name, data_type):
    log_format = '%(asctime)s - %(levelname)s -%(message)s'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(filename=os.path.join(os.getcwd(), f'exp_wav2vec2/reg/{data_type}/log.txt'), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(log_format))
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger

def set_config(data_type, batch_size, max_epochs):
    config = {
        "data_type": data_type,
        "seed": 308,
        "train_config": {
            "max_epochs": max_epochs,
            "audio_net": AUDIO_NET_CONFIG,
            "text_net": TEXT_NET_CONFIG,
            "fused_net": FUSED_NET_CONFIG,
            "optimizer": {
                "type": "adam",
                "lr": 8e-5, 
                "weight_decay": 0,
            }
        },
        "data_module": {
            "num_folds": 3,
            "data_type": data_type,
            "data_dir": os.path.join(os.getcwd(), "data/EATD-Feats/augment_wav2vec2"),
            "batch_size": batch_size
        }
    }
    return config

def set_data_module(config):
    # prepare EATD dataset
    eatd_dataset = EATDDataset(config["data_dir"], config["data_type"])
    # set up k-fold splits
    y = eatd_dataset.y
    X = np.zeros(y.shape[0])
    splits = [split for split in StratifiedKFold(n_splits=config["num_folds"], shuffle=True).split(X, y)]
    return eatd_dataset, splits
    
if __name__ == '__main__':
    data_type = "fuse"
    audio_ckp = os.path.join(os.getcwd(), "/home/dasein/Projects/Depression/exp_wav2vec2/reg/audio/fold0/model.fold0.epoch297.ckp")
    text_ckp = os.path.join(os.getcwd(), "/home/dasein/Projects/Depression/exp/reg/text/fold1/model.fold1.epoch276.ckp")
    
    batch_size = 8
    max_epochs = 300
    device = "cpu"
    logger = set_logger('reg', data_type)
    config = set_config(data_type, batch_size, max_epochs)
    dataset, splits = set_data_module(config["data_module"])
    
    loss_fn = RegFusedLoss(pos=FUSED_NET_CONFIG['audio_hidden_dim'])
    fold_results = {}
    # Start print
    logger.info('--------------------------------')
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(splits):
        logger.info(f"FOLD {fold}")
        logger.info('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=config["data_module"]["batch_size"], sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(test_ids), sampler=test_subsampler)
        
        
        audio_model = AudioBiLSTMNet(**config["train_config"]["audio_net"])
        audio_model.load_state_dict(torch.load(audio_ckp))
        audio_model = audio_model.to(device)
        
        text_model = TextBiLSTMNet(**config["train_config"]["text_net"])
        text_model.load_state_dict(torch.load(text_ckp))
        text_model = text_model.to(device)
        fused_model = FusedNet(**config["train_config"]["fused_net"])
        fused_model = fused_model.to(device)
        
        # Init optimizer
        if config["train_config"]["optimizer"]["type"] == "adamw":
            optimizer = torch.optim.AdamW(fused_model.parameters(), lr=config["train_config"]["optimizer"]["lr"], weight_decay=config["train_config"]["optimizer"]["weight_decay"])
        elif config["train_config"]["optimizer"]["type"] == "adam":
            optimizer = torch.optim.Adam(fused_model.parameters(), lr=config["train_config"]["optimizer"]["lr"], weight_decay=config["train_config"]["optimizer"]["weight_decay"])
            
        audio_model.train()
        for k, v in audio_model.named_parameters():
            v.requires_grad = False
        text_model.train()
        for k, v in text_model.named_parameters():
            v.requires_grad = False
        fused_model.train()
        
        best_mae, best_rmse = float('inf'), float('inf')
        best_epoch = -1
        # Run the training loop for defined number of epochs
        for epoch in range(0, max_epochs):
            tot_y_pred = np.array([])
            tot_y_true = np.array([])
            tot_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                y_audio, y_text, y_true, _ = data
                y_audio = y_audio.to(device)
                y_text = y_text.to(device)
                y_true = y_true.to(device)
                
                _, y_embed_audio = audio_model(y_audio)
                _, y_embed_text = text_model(y_text)
                y = torch.cat((y_embed_audio, y_embed_text), dim=1)
                
                optimizer.zero_grad()
                y_pred = fused_model(y)
                loss = loss_fn(y_embed_audio, y_embed_text, y_true, fused_model)
                loss.backward()
                optimizer.step()
                
                tot_y_pred = np.hstack((tot_y_pred, y_pred.flatten().detach().numpy()))
                tot_y_true = np.hstack((tot_y_true, y_true.detach().numpy()))
                tot_loss += loss.item()
            mae = mean_absolute_error(tot_y_true, tot_y_pred)
            rmse = np.sqrt(mean_squared_error(tot_y_true, tot_y_pred))
            if mae <= best_mae and rmse <= best_rmse:
                ckp_path = os.path.join(os.getcwd(), f'exp_wav2vec2/reg/{data_type}/fold{fold}/model.fold{fold}.epoch{epoch}.ckp')
                torch.save(fused_model.state_dict(), ckp_path)
                best_mae = min(mae, best_mae)
                best_rmse = min(rmse, best_rmse)
                best_epoch = epoch
            logger.debug("Training epoch:{:3d}\t loss:{:.6f}\t best_mae:{:.4f}\t best_rmse:{:.4f} best_epoch:{:3d}"
                  .format(epoch+1, tot_loss, best_mae, best_rmse, best_epoch))

        # logger.info('Training process has finished. Saving trained model.')
        # ckp_path = os.path.join(os.getcwd(), f'exp_wav2vec2/reg/{data_type}/model.fold{fold}.ckp')
        # torch.save(fused_model.state_dict(), ckp_path)
        
        # Evaluationfor this fold
        logger.info('Starting testing')
        best_ckp = os.path.join(os.getcwd(), f'exp_wav2vec2/reg/{data_type}/fold{fold}/model.fold{fold}.epoch{best_epoch}.ckp')
        audio_model.eval()
        text_model.eval()
        fused_model.load_state_dict(torch.load(best_ckp))
        fused_model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                y_audio, y_text, y_true, _ = data
                y_audio = y_audio.to(device)
                y_text = y_text.to(device)
                y_true = y_true.to(device)
                
                _, y_embed_audio = audio_model(y_audio)
                _, y_embed_text = text_model(y_text)
                y = torch.cat((y_embed_audio, y_embed_text), dim=1)
                
                y_pred = fused_model(y)
                loss = loss_fn(y_embed_audio, y_embed_text, y_true, fused_model)
                
                mae = mean_absolute_error(y_pred.flatten().detach().numpy(), y_true.detach().numpy())
                rmse = np.sqrt(mean_squared_error(y_pred.flatten().detach().numpy(), y_true.detach().numpy()))
                
                logger.info(f"======== Fold{fold}-Summary =========")
                logger.info("MAE : {:.4f}".format(mae))
                logger.info("RMSE: {:.4f}".format(rmse))
                logger.info("Best checkpoint: {}".format(best_ckp))
                logger.info("================================")
                fold_results[fold] = [mae, rmse]
    # Print fold results
    logger.info('K-FOLD CROSS VALIDATION RESULTS FOR {} FOLDS'.format(config["data_module"]["num_folds"]))
    logger.info('--------------------------------')
    avg_fold_results = np.sum([value for key, value in fold_results.items()], axis=0) / config["data_module"]["num_folds"]
    
    logger.info("MAE : {:.4f}".format(avg_fold_results[0]))
    logger.info("RMSE: {:.4f}".format(avg_fold_results[1]))
