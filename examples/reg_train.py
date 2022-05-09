import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from src.datamodules.eatd_dataset import EATDDataset
from src.models.regression.audio_bilstm_net import AudioBiLSTMNet
from src.models.regression.text_bilstm_net import TextBiLSTMNet
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


def set_config(data_type, batch_size, max_epochs):
    config = {
        "data_type": data_type,
        "seed": 308,
        "train_config": {
            "max_epochs": max_epochs,
            "net": AUDIO_NET_CONFIG if data_type == "audio" else TEXT_NET_CONFIG,
            "optimizer": {
                "type": "adam",
                "lr": 1e-5, 
                "weight_decay": 0,
            }
        },
        "data_module": {
            "num_folds": 3,
            "data_type": data_type,
            "data_dir": os.path.join(os.getcwd(), "data/EATD-Feats/augment"),
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
    data_type = "audio"
    batch_size = 8
    max_epochs = 150
    device = "cpu"
    
    config = set_config(data_type, batch_size, max_epochs)
    dataset, splits = set_data_module(config["data_module"])
    
    fold_results = {}
    # Start print
    print('--------------------------------')
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(splits):
        print(f"FOLD {fold}")
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=config["data_module"]["batch_size"], sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(test_ids), sampler=test_subsampler)
        
        # Init the neural network model
        if config["data_type"] == "audio":
            model = AudioBiLSTMNet(**config["train_config"]["net"])
        elif config["data_type"] == "text":
            model = TextBiLSTMNet(**config["train_config"]["net"])
        model = model.to(device)
        
        # Init optimizer
        if config["train_config"]["optimizer"]["type"] == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["train_config"]["optimizer"]["lr"], weight_decay=config["train_config"]["optimizer"]["weight_decay"])
        elif config["train_config"]["optimizer"]["type"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["train_config"]["optimizer"]["lr"], weight_decay=config["train_config"]["optimizer"]["weight_decay"])
            
        model.train()
        
        # Run the training loop for defined number of epochs
        for epoch in range(0, max_epochs):
            tot_y_pred = np.array([])
            tot_y_true = np.array([])
            tot_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                y, y_true, _ = data
                y = y.to(device)
                y_true = y_true.to(device)
                
                optimizer.zero_grad()
                y_pred = model(y)
                
                loss = F.smooth_l1_loss(y_pred, y_true.view_as(y_pred))
                loss.backward()
                optimizer.step()
                
                tot_y_pred = np.hstack((tot_y_pred, y_pred.flatten().detach().numpy()))
                tot_y_true = np.hstack((tot_y_true, y_true.detach().numpy()))
                tot_loss += loss.item()
            mae = mean_absolute_error(tot_y_true, tot_y_pred)
            rmse = np.sqrt(mean_squared_error(tot_y_true, tot_y_pred))
            print("Training epoch:{:3d}\t loss:{:.6f}\t mae:{:.4f}\t rmse:{:.4f}"
                  .format(epoch+1, tot_loss, mae, rmse))

        print('Training process has finished. Saving trained model.')
        ckp_path = os.path.join(os.getcwd(), f'exp/reg/{data_type}/model.fold{fold}.ckp')
        torch.save(model.state_dict(), ckp_path)
        
        # Evaluationfor this fold
        print('Starting testing')
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                y, y_true, _ = data
                y = y.to(device)
                y_true = y_true.to(device)
                
                y_pred = model(y)
                loss = F.smooth_l1_loss(y_pred, y_true.view_as(y_pred))
                
                
                mae = mean_absolute_error(y_pred.flatten().detach().numpy(), y_true.detach().numpy())
                rmse = np.sqrt(mean_squared_error(y_pred.flatten().detach().numpy(), y_true.detach().numpy()))
                
                print(f"======== Fold{fold}-Summary =========")
                print("MAE : {:.4f}".format(mae))
                print("RMSE: {:.4f}".format(rmse))
                print("================================")
                fold_results[fold] = [mae, rmse]
    # Print fold results
    print('K-FOLD CROSS VALIDATION RESULTS FOR {} FOLDS'.format(config["data_module"]["num_folds"]))
    print('--------------------------------')
    avg_fold_results = np.sum([value for key, value in fold_results.items()], axis=0) / config["data_module"]["num_folds"]
    
    print("MAE : {:.4f}".format(avg_fold_results[0]))
    print("RMSE: {:.4f}".format(avg_fold_results[1]))
