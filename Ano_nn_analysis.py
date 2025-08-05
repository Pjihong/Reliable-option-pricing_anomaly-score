import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt

import math
import torch.nn.functional as F
from tqdm import tqdm


import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_dim):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 1)
               
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, start_epoch=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.start_epoch = start_epoch

    def __call__(self, val_loss, epoch):
        if epoch < self.start_epoch:
            return

        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def loop(model, dataloader, optimizer=None, rank=None, k=0.0, use_weight=True):
    is_eval_mode_for_loop = not optimizer 

    if is_eval_mode_for_loop:
        torch.set_grad_enabled(False)
    
    losses = []
    for batch_idx, (x_sample, y_sample, *args) in enumerate(dataloader):
        x_sample = x_sample.to(rank)
        y_sample = y_sample.to(rank)
        
        if len(args) > 0 and use_weight:
            ano_score = torch.exp(-args[0].to(rank) * k)
        else:
            ano_score = torch.ones_like(y_sample)
            
        if optimizer:
            optimizer.zero_grad()

        pred = model(x_sample)
        loss = torch.sum((pred - y_sample) ** 2 * ano_score) 
        loss /= x_sample.size(0)
        losses.append(loss.item())
                
        if optimizer:
            loss.backward()
            optimizer.step()
            
    if is_eval_mode_for_loop:
        torch.set_grad_enabled(True)
        
    return np.mean(losses)


def train_and_validate_model(k_value,
                             x_train,
                             dataloader_train,
                             dataloader_train_eval,
                             dataloader_val_eval,
                             rank,
                             EPOCHS=200,
                             patience=20):
    model = NN(x_train.shape[1]).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    start_epoch = int(EPOCHS * 0.75)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, start_epoch=start_epoch)
    
    loss_list = np.zeros((EPOCHS, 2))
    
    report_epochs = [e for e in [100, 200, 300, 400, 500, 600, 700, 800] if e <= EPOCHS]
    if not any(ep == EPOCHS for ep in report_epochs) and EPOCHS > 0 and EPOCHS < report_epochs[0] if report_epochs else True : 
         if EPOCHS not in report_epochs : report_epochs.append(EPOCHS) # Add last epoch
         report_epochs.sort()

    min_losses_at_epochs = {
        epoch_milestone: {'train': float('inf'), 'val': float('inf')}
        for epoch_milestone in report_epochs
    }
    if not min_losses_at_epochs and EPOCHS > 0 : 
        min_losses_at_epochs[EPOCHS] = {'train': float('inf'), 'val': float('inf')}


    last_epoch = 0
    
    print(f"Early stopping will start monitoring from epoch {start_epoch + 1}")
    pbar = tqdm(range(EPOCHS), desc=f"Training k={k_value}", leave=False)
    
    for epoch in pbar:
        # ---- ① Actual training step ----
        model.train()
        loop(model, dataloader_train, optimizer, rank=rank, k=k_value, use_weight=True)
        
        # ---- ② Evaluation step ----
        model.train() 
        train_mse_noisy = loop(model, dataloader_train_eval, optimizer=None, rank=rank, k=0.0, use_weight=False)
        
        model.eval() 
        train_mse_clean = loop(model, dataloader_train_eval, optimizer=None, rank=rank, k=0.0, use_weight=False)
        
        val_mse_clean = loop(model, dataloader_val_eval, optimizer=None, rank=rank, k=0.0, use_weight=False)
        
        loss_list[epoch, 0] = train_mse_noisy
        loss_list[epoch, 1] = val_mse_clean
        
        last_epoch = epoch
        
        pbar.set_postfix({
            "TrainMSE(noisy)": f"{train_mse_noisy:.4f}",
            "TrainMSE(clean)": f"{train_mse_clean:.4f}", 
            "ValMSE(clean)": f"{val_mse_clean:.4f}",
        })
        
        early_stopping(val_mse_clean, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1} (patience={patience}).")
            # Fill remaining loss_list after stopping (for consistent graphs)
            loss_list[epoch+1:, 0] = train_mse_noisy 
            loss_list[epoch+1:, 1] = val_mse_clean
            break
            
    final_recorded_epoch_count = last_epoch + 1
    
    for target_epoch_val in min_losses_at_epochs.keys():
        effective_epochs = min(target_epoch_val, final_recorded_epoch_count)
        if effective_epochs > 0 : # Calculate only when there are valid epochs
            min_losses_at_epochs[target_epoch_val]['train'] = np.min(loss_list[:effective_epochs, 0])
            min_losses_at_epochs[target_epoch_val]['val'] = np.min(loss_list[:effective_epochs, 1])

    min_val_mse_overall = early_stopping.best_loss if early_stopping.best_loss is not None else float('inf')
    
    return loss_list, min_val_mse_overall, min_losses_at_epochs, last_epoch

def evaluate_model_single(model, dataloader, rank):
    model.eval()
    test_pred = np.array([])
    y_result = np.array([])

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x_sample, y_sample, _ = batch
            else:
                x_sample, y_sample = batch  
            x_sample = x_sample.to(rank)
            y_sample = y_sample.to(rank)
            pred = model(x_sample)

            if len(test_pred) == 0:
                print("\n[DEBUG] First batch predictions:", pred[:10].detach().cpu().numpy().flatten())
                print("[DEBUG] First batch actual values:", y_sample[:10].detach().cpu().numpy().flatten())

            test_pred = np.concatenate((test_pred, pred.detach().cpu().numpy().flatten()))  
            y_result = np.concatenate((y_result, y_sample.detach().cpu().numpy().flatten()))
    
    diff = test_pred - y_result
    squared_diff = diff ** 2

    total_mse = np.mean(squared_diff)

    return total_mse
    

def evaluate_model(model, dataloader_test, test_dataloaders, rank):
    model.eval()
    total_mse = evaluate_model_single(model, dataloader_test, rank)
    mse_values = evaluate_model_set(model, test_dataloaders, rank)
    mse_values.insert(0, total_mse)
    return mse_values


def predict_and_store_results(dataloader, model, device):
    test_pred = np.array([])
    y_result = np.array([])

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x_sample, y_sample, _ = batch
            else:
                x_sample, y_sample = batch
            x_sample = x_sample.to(device)
            pred = model(x_sample).detach().cpu().numpy()
            test_pred = np.concatenate((test_pred, pred.flatten()))
            y_result = np.concatenate((y_result, y_sample.detach().cpu().numpy().flatten()))

    return test_pred, y_result
    

def evaluate_model_set(model, dataloaders, rank):
    model.eval()
    mse_values = []
    for idx, dataloader in enumerate(dataloaders, 1):
        test_pred, y_result = predict_and_store_results(dataloader, model, rank)
        diff = test_pred - y_result.flatten()
        squared_diff = diff ** 2
        mse = np.mean(squared_diff)
        mse_values.append(mse)
    return mse_values


def check_maturity(data):
    data_check = data[['today', 'maturity_date', 'remaining_maturity']]
    max_diff = (data_check['maturity_date'] - data_check['today']).max()
    max_remaining_maturity = data_check['remaining_maturity'].max()
    return max_diff, max_remaining_maturity


def option_type(row):
    moneyness = row['simple moneyness']
    if moneyness < 0.95:
        return 'ITM (Put)'
    elif moneyness > 1.05:
        return 'OTM (Put)'
    else:
        return 'ATM (Put)'


def simple_classify(row):
    if 'ITM' in row:
        return 'ITM'
    elif 'OTM' in row:
        return 'OTM'
    else:
        return 'ATM'


def classify_expiry(row):
    if row < 0.04165:
        return 'biweek'
    elif 0.04165 <= row < 0.0825: 
        return 'onemonth'
    elif 0.0825 <= row < 0.25:
        return 'quarter'
    elif 0.25 <= row < 0.5:
        return 'half'
    else:
        return 'year'
    

def create_dataloader(data, y_test):
    dataset = TensorDataset(torch.from_numpy(data).float(), torch.from_numpy(y_test).float())
    dataloader = DataLoader(dataset, shuffle=False, num_workers=8, drop_last=True)
    return dataloader

def apply_and_save_splits(data, time_splits):
    for idx, (train_start, train_end, test_start, test_end) in enumerate(time_splits):
        data_train_val = data[(data['today'] >= train_start) & (data['today'] <= train_end)].reset_index(drop=True)
        data_test = data[(data['today'] >= test_start) & (data['today'] <= test_end)].reset_index(drop=True)
        
        train_file_name = f'krx_train_data_{train_start}_to_{train_end}.pkl'
        test_file_name = f'krx_test_data_{test_start}_to_{test_end}.pkl'
        
        train_file_path = f'data/{train_file_name}'
        test_file_path = f'data/{test_file_name}'
        
        data_train_val.to_pickle(train_file_path)
        data_test.to_pickle(test_file_path)
        
        print(f'Train data saved: {train_file_name}, Test data saved: {test_file_name}')


def visualize_and_print_stats(loss_list, k_value, min_losses_at_epochs, last_epoch):
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(10, 5))
    plt.title(f"Session Training and Validation MSE, k={k_value}", fontsize=16)
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    start_epoch_plot = 100
    epochs_to_plot = np.arange(start_epoch_plot, len(loss_list))

    if len(loss_list) > start_epoch_plot:
        plt.plot(epochs_to_plot, loss_list[start_epoch_plot:, 1], 'b--', linewidth=2, label='Validation MSE')
        plt.plot(epochs_to_plot, loss_list[start_epoch_plot:, 0], 'r-', linewidth=1, label='Train MSE')
    else:
        plt.plot(loss_list[:, 1], 'b--', linewidth=2, label='Validation MSE')
        plt.plot(loss_list[:, 0], 'r-', linewidth=2, label='Train MSE')

    plt.legend(fontsize=12) # Show legend
    plt.grid(True, linestyle='--', alpha=0.7) # Change grid style
    plt.show()

    min_train_mse = np.min(loss_list[:, 0])
    min_val_mse = np.min(loss_list[:, 1])
    print(f"--- Stats for k={k_value} (Last Session) ---")
    print("Minimum Train MSE:", min_train_mse)
    print("Minimum Validation MSE:", min_val_mse)
    print("RMSE Train:", np.sqrt(min_train_mse))
    print("RMSE Validation:", np.sqrt(min_val_mse))
    
    print("\nMinimum loss values up to each epoch:")
    for target_epoch in [100, 200, 300]:
        status = "reached" if last_epoch >= target_epoch - 1 else "not reached"
        print(f"Epoch {target_epoch} ({status}):")
        if np.isfinite(min_losses_at_epochs[target_epoch]['train']):
            print(f"  Minimum Train MSE: {min_losses_at_epochs[target_epoch]['train']:.6f}")
            print(f"  Minimum Val MSE: {min_losses_at_epochs[target_epoch]['val']:.6f}")
        else:
            print(f"  No data (epoch {target_epoch} not reached)")


def plot_mse_comparison(mse_values_k0, mse_values_k7, labels):
    print("Length of mse_values_k0:", len(mse_values_k0))
    print("Length of mse_values_k7:", len(mse_values_k7))
    print("Length of labels:", len(labels))
    x = np.arange(len(labels)) 
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    rects1 = ax1.bar(x - width/2, mse_values_k0, width, label='K=0.0', color='blue')
    rects2 = ax1.bar(x + width/2, mse_values_k7, width, label='K=0.7', color='green')

    ax1.set_xlabel('Option Type', fontsize=15)
    ax1.set_ylabel('MSE', fontsize=15)
    ax1.set_title('Comparison of MSE for K=0.0 and K=0.7', fontsize=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=15)
    ax1.legend(fontsize=15)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate('{}'.format(round(height, 4)),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), 
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=12)
    for i, (k0, k7) in enumerate(zip(mse_values_k0, mse_values_k7)):
        if k0 != 0:
            relative_error = abs((k7 - k0) / k0)
            ax1.annotate('Rel. Error: {:.2%}'.format(relative_error),
                         xy=(x[i], max(mse_values_k0[i], mse_values_k7[i])),
                         xytext=(0, 10), 
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=12, color='purple')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    

def plot_train_loss(loss_list):
    fig = plt.figure(figsize=(8, 8))
    plt.ylabel('MSE', fontsize=15)
    plt.xlabel('epoch', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(loss_list[:, 0], 'r', linewidth=2, label='train')
    plt.plot(loss_list[:, 1], 'b', linewidth=2, label='val')
    plt.legend(fontsize=15)
    plt.title('Training and Validation Loss', fontsize=15)
    plt.show()

    print(f"Minimum Training MSE: {loss_list[:, 0].min():.4f}")
    print(f"Minimum Validation MSE: {loss_list[:, 1].min():.4f}")


def plot_test_losses_with_std(test_losses, titles, k_values, ylims):
    fig, axes = plt.subplots(nrows=len(test_losses), figsize=(10, 6 * len(test_losses)))
    if len(test_losses) == 1:
        axes = [axes]  
    for i, losses in enumerate(test_losses):
        median_test_losses = [np.median(session) for session in losses]
        std_test_losses = [np.std(session) for session in losses]
        for j in range(len(losses)):
            axes[i].scatter([k_values[j]] * len(losses[j]), losses[j], color='blue', alpha=0.6)
        
        axes[i].errorbar(k_values, median_test_losses, yerr=std_test_losses, fmt='o', color='red', label='Median Test Loss', zorder=5)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('K Value')
        axes[i].set_ylabel('Test Loss')
        axes[i].legend()
        axes[i].grid(True)
        if ylims[i]:
            axes[i].set_ylim(ylims[i])
    plt.tight_layout()
    plt.show()


def plot_test_losses_with_error_bars(test_losses, titles, k_values, ylims):
    fig, axes = plt.subplots(nrows=len(test_losses), figsize=(10, 6 * len(test_losses)))
    if len(test_losses) == 1:
        axes = [axes]  # Ensure axes is iterable even for a single subplot
    for i, losses in enumerate(test_losses):
        median_test_losses = [np.median(session) for session in losses]
        std_test_losses = [np.std(session) for session in losses]
        for j in range(len(losses)):
            axes[i].scatter([k_values[j]] * len(losses[j]), losses[j], color='blue', alpha=0.6)
        
        axes[i].errorbar(k_values, median_test_losses, yerr=std_test_losses, fmt='o', color='red', label='Median Test Loss', zorder=5)
        
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('K Value')
        axes[i].set_ylabel('Test Loss')
        axes[i].legend()
        axes[i].grid(True)
        if ylims[i]:
            axes[i].set_ylim(ylims[i])
    plt.tight_layout()
    plt.show()


def plot_test_losses_plot(test_losses, titles, k_values, ylims):
    fig, axes = plt.subplots(nrows=len(test_losses), figsize=(10, 6 * len(test_losses)))
    if len(test_losses) == 1:
        axes = [axes] 
    for i, losses in enumerate(test_losses):
        median_test_losses = [np.median(session) for session in losses]
        std_test_losses = [np.std(session) for session in losses]
        
        for j in range(len(losses)):
            axes[i].scatter([k_values[j]] * len(losses[j]), losses[j], color='blue', alpha=0.6)
        
        axes[i].scatter(k_values, median_test_losses, color='red', label='Median Test Loss', zorder=5)
        axes[i].fill_between(k_values, 
                             [m - s for m, s in zip(median_test_losses, std_test_losses)], 
                             [m + s for m, s in zip(median_test_losses, std_test_losses)], 
                             color='lightblue', alpha=0.5, label='Standard Deviation')
        
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('K Value')
        axes[i].set_ylabel('Test Loss')
        axes[i].legend()
        axes[i].grid(True)
        if ylims[i]:
            axes[i].set_ylim(ylims[i])
    plt.tight_layout()
    plt.show()

time_splits = [
    ('2019-01-01', '2019-12-31', '2020-01-01', '2020-06-30'),
    ('2019-07-01', '2020-06-30', '2020-07-01', '2020-12-31'),
    ('2020-01-01', '2020-12-31', '2021-01-01', '2021-06-30'),
    ('2020-07-01', '2021-06-30', '2021-07-01', '2021-12-31'),
    ('2021-01-01', '2021-12-31', '2022-01-01', '2022-06-30'),
    ('2021-07-01', '2022-06-30', '2022-07-01', '2022-12-31'),
    ('2022-01-01', '2022-12-31', '2023-01-01', '2023-06-30'),
    ('2022-07-01', '2023-06-30', '2023-07-01', '2023-12-31')
]
