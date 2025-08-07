import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

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
    is_eval_mode_for_loop = not optimizer # optimizer가 없으면 평가 목적의 호출로 간주

    # 모델의 원래 학습 상태 저장 (loop 함수 진입 시점)
    # model.train() 또는 model.eval()은 loop 함수 외부에서 호출자가 설정해야 함.
    # loop 함수는 그 상태를 존중하되, gradient 계산만 제어.
    
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
                             x_train,                # 모델 input_dim 결정을 위해 필요
                             dataloader_train,       # 실제 학습용 (shuffle=True, anomaly weight 사용 가능)
                             dataloader_train_eval,  # Train MSE 평가용 (shuffle=False, anomaly weight 미사용)
                             dataloader_val_eval,    # Validation MSE 평가용 (shuffle=False, anomaly weight 미사용)
                             rank,
                             EPOCHS=200,
                             patience=20):
    model = NN(x_train.shape[1]).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    start_epoch = int(EPOCHS * 0.75)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, start_epoch=start_epoch)
    
    # loss_list 컬럼:
    # 0: Train MSE (Dropout ON, unweighted) - 노이즈 있는 학습 곡선용
    # 1: Validation MSE (Dropout OFF, unweighted)
    loss_list = np.zeros((EPOCHS, 2))

    last_epoch = 0
    
    print(f"Early stopping will start monitoring from epoch {start_epoch + 1}")
    pbar = tqdm(range(EPOCHS), desc=f"Training k={k_value}", leave=False)
    
    for epoch in pbar:
        # ---- ① 실제 학습 단계 (Optimizer 사용, Dropout ON, Anomaly Weight 사용) ----
        model.train()
        # 아래 loop의 반환값은 anomaly score로 가중된 학습 손실이므로, 직접 사용하지 않음.
        # 시각화를 위한 "순수" MSE (가중치 미적용)는 아래에서 별도 계산.
        loop(model, dataloader_train, optimizer, rank=rank, k=k_value, use_weight=True)
        
        # ---- ② 평가 단계 ----
        # (A) 학습용 MSE 계산 (Dropout ON, 가중치 미적용) - "노이즈 있는" 학습 곡선용
        model.train() # Dropout, BatchNorm 등을 학습 상태로 유지
        train_mse_noisy = loop(model, dataloader_train_eval, optimizer=None, rank=rank, k=0.0, use_weight=False)
        
        # (B) 학습용 MSE 계산 (Dropout OFF, 가중치 미적용) - "깨끗한" 학습 성능 참고용
        model.eval() # Dropout, BatchNorm 등을 평가 상태로 변경
        train_mse_clean = loop(model, dataloader_train_eval, optimizer=None, rank=rank, k=0.0, use_weight=False)
        
        # (C) 검증용 MSE 계산 (Dropout OFF, 가중치 미적용)
        # model.eval() # 이미 평가 상태
        val_mse_clean = loop(model, dataloader_val_eval, optimizer=None, rank=rank, k=0.0, use_weight=False)
        
        # loss_list에 기록 (시각화용)
        loss_list[epoch, 0] = train_mse_noisy
        loss_list[epoch, 1] = val_mse_clean
        
        last_epoch = epoch
        
        pbar.set_postfix({
            "TrainMSE(noisy)": f"{train_mse_noisy:.4f}",
            "TrainMSE(clean)": f"{train_mse_clean:.4f}", # 참고용으로 출력
            "ValMSE(clean)": f"{val_mse_clean:.4f}",
        })
        
        # Early Stopping은 깨끗한 Validation MSE를 사용
        early_stopping(val_mse_clean, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1} (patience={patience}).")
            # 중단 후 나머지 loss_list 채우기 (일관된 그래프를 위해)
            loss_list[epoch+1:, 0] = train_mse_noisy 
            loss_list[epoch+1:, 1] = val_mse_clean
            break
    
    # 전체 기간 중 가장 좋았던 (EarlyStopping에 의해 선택된) Validation MSE
    min_val_mse_overall = early_stopping.best_loss if early_stopping.best_loss is not None else float('inf')
    
    return loss_list, min_val_mse_overall, last_epoch

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

            # 디버깅: 첫 번째 배치만 예측 결과 출력
            if len(test_pred) == 0:
                print("\n[DEBUG] 첫 배치 예측값:", pred[:10].detach().cpu().numpy().flatten())
                print("[DEBUG] 첫 배치 실제값:", y_sample[:10].detach().cpu().numpy().flatten())

            test_pred = np.concatenate((test_pred, pred.detach().cpu().numpy().flatten()))  
            y_result = np.concatenate((y_result, y_sample.detach().cpu().numpy().flatten()))
    
    # 차이 및 제곱차이 확인
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


def check_만기(data):
    data_check = data[['today', '만기일', '잔존만기']]
    max_diff = (data_check['만기일'] - data_check['today']).max()
    max_잔존만기 = data_check['잔존만기'].max()
    return max_diff, max_잔존만기


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
    elif 0.04165 <= row < 0.0825:  # ~1 month
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


# 시간 분할을 적용하고 데이터를 저장하는 함수
def apply_and_save_splits(data, time_splits):
    for idx, (train_start, train_end, test_start, test_end) in enumerate(time_splits):
        # 훈련 및 테스트 데이터 분할
        data_train_val = data[(data['today'] >= train_start) & (data['today'] <= train_end)].reset_index(drop=True)
        data_test = data[(data['today'] >= test_start) & (data['today'] <= test_end)].reset_index(drop=True)
        
        # 파일 이름 정의
        train_file_name = f'krx_train_data_{train_start}_to_{train_end}.pkl'
        test_file_name = f'krx_test_data_{test_start}_to_{test_end}.pkl'
        
        # 파일 경로 설정
        train_file_path = f'data/{train_file_name}'
        test_file_path = f'data/{test_file_name}'
        
        # 데이터 저장
        data_train_val.to_pickle(train_file_path)
        data_test.to_pickle(test_file_path)
        
        # 저장 상태 출력
        print(f'Train data saved: {train_file_name}, Test data saved: {test_file_name}')


def visualize_and_print_stats(loss_list, k_value, last_epoch):
    # 한글 폰트 설정
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(10, 5)) # 그래프 크기 약간 조정
    plt.title(f"Session Training and Validation MSE, k={k_value}", fontsize=16)
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 전체 에폭의 1/4 지점부터 시작하도록 변경
    start_epoch_plot = max(1, len(loss_list) // 4)
    epochs_to_plot = np.arange(start_epoch_plot, len(loss_list))

    if len(loss_list) > start_epoch_plot:
        # Validation MSE를 먼저 그리고, Train MSE를 그 위에 그림 (스타일 변경)
        plt.plot(epochs_to_plot, loss_list[start_epoch_plot:, 1], 'b--', linewidth=2, label='Validation MSE') # 파란색 점선
        plt.plot(epochs_to_plot, loss_list[start_epoch_plot:, 0], 'r-', linewidth=1, label='Train MSE')      # 빨간색 실선
    else:
        # Epoch 수가 적을 경우 전체를 그림
        plt.plot(loss_list[:, 1], 'b--', linewidth=2, label='Validation MSE')
        plt.plot(loss_list[:, 0], 'r-', linewidth=2, label='Train MSE')

    plt.legend(fontsize=12) # 범례 표시
    plt.grid(True, linestyle='--', alpha=0.7) # 그리드 스타일 변경
    plt.show()

    min_train_mse = np.min(loss_list[:, 0])
    min_val_mse = np.min(loss_list[:, 1])
    print(f"--- Stats for k={k_value} (Last Session) ---")
    print("Minimum Train MSE:", min_train_mse)
    print("Minimum Validation MSE:", min_val_mse)
    print("RMSE Train:", np.sqrt(min_train_mse))
    print("RMSE Validation:", np.sqrt(min_val_mse))
    print(f"Training completed at epoch: {last_epoch + 1}")


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
        axes = [axes]  # Ensure axes is iterable even for a single subplot
    for i, losses in enumerate(test_losses):
        median_test_losses = [np.median(session) for session in losses]
        std_test_losses = [np.std(session) for session in losses]
        for j in range(len(losses)):
            axes[i].scatter([k_values[j]] * len(losses[j]), losses[j], color='blue', alpha=0.6)
        
        # Plot median test losses with error bars for standard deviation
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
        
        # Plot median test losses with error bars for standard deviation
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
        axes = [axes]  # Ensure axes is iterable even for a single subplot
    for i, losses in enumerate(test_losses):
        median_test_losses = [np.median(session) for session in losses]
        std_test_losses = [np.std(session) for session in losses]
        
        # Plot individual points
        for j in range(len(losses)):
            axes[i].scatter([k_values[j]] * len(losses[j]), losses[j], color='blue', alpha=0.6)
        
        # Plot median points
        axes[i].scatter(k_values, median_test_losses, color='red', label='Median Test Loss', zorder=5)
        
        # Plot ribbon for standard deviation
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