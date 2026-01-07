"""
MLP Binary Classification - Advanced Version
進階版本：包含更多視覺化、統計分析、特徵重要性、錯誤分析等
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')

# 設定隨機種子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# CUDA 性能優化（針對 RTX A6000）
torch.backends.cudnn.deterministic = False  # 啟用性能優化
torch.backends.cudnn.benchmark = True       # 自動調優卷積算法
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True   # 啟用 TensorFloat-32
    torch.backends.cudnn.allow_tf32 = True         # A6000 專用加速

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 創建輸出目錄
os.makedirs('results', exist_ok=True)
os.makedirs('results/folds', exist_ok=True)
os.makedirs('results/statistics', exist_ok=True)
os.makedirs('results/advanced', exist_ok=True)


class MLP_Dataset(Dataset):
    """PyTorch Dataset"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ParametricTanh(nn.Module):
    """可学习参数的Tanh激活函数: output = alpha * tanh(beta * x) + gamma"""
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # 缩放参数
        self.beta = nn.Parameter(torch.tensor(beta))    # 输入缩放
        self.gamma = nn.Parameter(torch.tensor(gamma))  # 偏移参数
    
    def forward(self, x):
        return self.alpha * torch.tanh(self.beta * x) + self.gamma


class ParametricSigmoid(nn.Module):
    """可学习参数的Sigmoid激活函数: output = alpha * sigmoid(beta * x) + gamma"""
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # 缩放参数
        self.beta = nn.Parameter(torch.tensor(beta))    # 输入缩放（控制陡峭度）
        self.gamma = nn.Parameter(torch.tensor(gamma))  # 偏移参数
    
    def forward(self, x):
        return self.alpha * torch.sigmoid(self.beta * x) + self.gamma


class EarlyStopping:
    """Early Stopping with Warmup"""
    def __init__(self, patience=30, start_delay=50, mode='min'):
        self.patience = patience
        self.start_delay = start_delay
        self.counter = 0
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.mode = mode
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if epoch < self.start_delay:
            return False
        improved = score < self.best_score if self.mode == 'min' else score > self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ShallowMLP(nn.Module):
    """淺層 MLP: 5層 - 160 → 128 → 96 → 64 → 32 → 1"""
    def __init__(self, use_parametric_activation=False):
        super().__init__()
        self.use_parametric = use_parametric_activation
        self.fc1 = nn.Linear(160, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
        # 參數化激活函數（前3層使用）
        if use_parametric_activation:
            self.act1 = ParametricTanh(alpha=1.5, beta=1.2, gamma=0.1)
            self.act2 = ParametricSigmoid(alpha=1.8, beta=1.5, gamma=-0.3)
            self.act3 = ParametricTanh(alpha=1.3, beta=1.1, gamma=0.05)
        
        self.activation_outputs = {}
    
    def forward(self, x, return_embedding=False, record_activations=False):
        if self.use_parametric:
            x = self.dropout(self.act1(self.fc1(x)))
            if record_activations: self.activation_outputs['layer1'] = x.detach().cpu()
            x = self.dropout(self.act2(self.fc2(x)))
            if record_activations: self.activation_outputs['layer2'] = x.detach().cpu()
            x = self.dropout(self.act3(self.fc3(x)))
            if record_activations: self.activation_outputs['layer3'] = x.detach().cpu()
        else:
            x = self.dropout(torch.relu(self.fc1(x)))
            if record_activations: self.activation_outputs['layer1'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.fc2(x)))
            if record_activations: self.activation_outputs['layer2'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.fc3(x)))
            if record_activations: self.activation_outputs['layer3'] = x.detach().cpu()
        x = torch.relu(self.fc4(x))
        emb = x
        x = self.fc5(x)  # 移除 sigmoid（使用 BCEWithLogitsLoss）
        return (x, emb) if return_embedding else x


class MediumMLP(nn.Module):
    """中層 MLP: 10層 - 160 → 256 → 224 → 192 → 160 → 128 → 96 → 64 → 48 → 32 → 1"""
    def __init__(self, use_parametric_activation=False):
        super().__init__()
        self.use_parametric = use_parametric_activation
        self.fc1 = nn.Linear(160, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 224)
        self.bn2 = nn.BatchNorm1d(224)
        self.fc3 = nn.Linear(224, 192)
        self.bn3 = nn.BatchNorm1d(192)
        self.fc4 = nn.Linear(192, 160)
        self.bn4 = nn.BatchNorm1d(160)
        self.fc5 = nn.Linear(160, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 96)
        self.fc7 = nn.Linear(96, 64)
        self.fc8 = nn.Linear(64, 48)
        self.fc9 = nn.Linear(48, 32)
        self.fc10 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        
        # 參數化激活函數（前6層使用）
        if use_parametric_activation:
            self.act1 = ParametricTanh(alpha=1.5, beta=1.2, gamma=0.1)
            self.act2 = ParametricSigmoid(alpha=1.8, beta=1.5, gamma=-0.3)
            self.act3 = ParametricTanh(alpha=1.3, beta=1.1, gamma=0.05)
            self.act4 = ParametricSigmoid(alpha=1.6, beta=1.3, gamma=-0.25)
            self.act5 = ParametricTanh(alpha=1.2, beta=1.0, gamma=0.08)
            self.act6 = ParametricSigmoid(alpha=1.7, beta=1.4, gamma=-0.35)
        
        self.activation_outputs = {}
    
    def forward(self, x, return_embedding=False, record_activations=False):
        if self.use_parametric:
            x = self.dropout(self.act1(self.bn1(self.fc1(x))))
            if record_activations: self.activation_outputs['layer1'] = x.detach().cpu()
            x = self.dropout(self.act2(self.bn2(self.fc2(x))))
            if record_activations: self.activation_outputs['layer2'] = x.detach().cpu()
            x = self.dropout(self.act3(self.bn3(self.fc3(x))))
            if record_activations: self.activation_outputs['layer3'] = x.detach().cpu()
            x = self.dropout(self.act4(self.bn4(self.fc4(x))))
            if record_activations: self.activation_outputs['layer4'] = x.detach().cpu()
            x = self.dropout(self.act5(self.bn5(self.fc5(x))))
            if record_activations: self.activation_outputs['layer5'] = x.detach().cpu()
            x = self.dropout(self.act6(self.fc6(x)))
            if record_activations: self.activation_outputs['layer6'] = x.detach().cpu()
        else:
            x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
            if record_activations: self.activation_outputs['layer1'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
            if record_activations: self.activation_outputs['layer2'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
            if record_activations: self.activation_outputs['layer3'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
            if record_activations: self.activation_outputs['layer4'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn5(self.fc5(x))))
            if record_activations: self.activation_outputs['layer5'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.fc6(x)))
            if record_activations: self.activation_outputs['layer6'] = x.detach().cpu()
        x = self.dropout(torch.relu(self.fc7(x)))
        x = self.dropout(torch.relu(self.fc8(x)))
        x = torch.relu(self.fc9(x))
        emb = x
        x = self.fc10(x)  # 移除 sigmoid（使用 BCEWithLogitsLoss）
        return (x, emb) if return_embedding else x


class DeepMLP(nn.Module):
    """深層 MLP: 20層 - 使用参数化激活函数"""
    def __init__(self, use_parametric_activation=False):
        super().__init__()
        self.use_parametric = use_parametric_activation
        
        self.fc1 = nn.Linear(160, 384)
        self.bn1 = nn.BatchNorm1d(384)
        self.fc2 = nn.Linear(384, 352)
        self.bn2 = nn.BatchNorm1d(352)
        self.fc3 = nn.Linear(352, 320)
        self.bn3 = nn.BatchNorm1d(320)
        self.fc4 = nn.Linear(320, 288)
        self.bn4 = nn.BatchNorm1d(288)
        self.fc5 = nn.Linear(288, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 224)
        self.bn6 = nn.BatchNorm1d(224)
        self.fc7 = nn.Linear(224, 192)
        self.bn7 = nn.BatchNorm1d(192)
        self.fc8 = nn.Linear(192, 160)
        self.bn8 = nn.BatchNorm1d(160)
        self.fc9 = nn.Linear(160, 128)
        self.bn9 = nn.BatchNorm1d(128)
        self.fc10 = nn.Linear(128, 96)
        self.bn10 = nn.BatchNorm1d(96)
        self.fc11 = nn.Linear(96, 80)
        self.bn11 = nn.BatchNorm1d(80)
        self.fc12 = nn.Linear(80, 64)
        self.bn12 = nn.BatchNorm1d(64)
        self.fc13 = nn.Linear(64, 48)
        self.fc14 = nn.Linear(48, 36)
        self.fc15 = nn.Linear(36, 28)
        self.fc16 = nn.Linear(28, 20)
        self.fc17 = nn.Linear(20, 16)
        self.fc18 = nn.Linear(16, 12)
        self.fc19 = nn.Linear(12, 8)
        self.fc20 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.5)
        
        # 参数化激活函数（交替使用Tanh和Sigmoid）- 前12层使用
        # 調整後的參數配置：優化alpha縮放、beta陡峭度、gamma偏移
        if use_parametric_activation:
            self.act1 = ParametricTanh(alpha=1.5, beta=1.2, gamma=0.1)
            self.act2 = ParametricSigmoid(alpha=1.8, beta=1.5, gamma=-0.3)
            self.act3 = ParametricTanh(alpha=1.3, beta=1.1, gamma=0.05)
            self.act4 = ParametricSigmoid(alpha=1.6, beta=1.3, gamma=-0.25)
            self.act5 = ParametricTanh(alpha=1.2, beta=1.0, gamma=0.08)
            self.act6 = ParametricSigmoid(alpha=1.7, beta=1.4, gamma=-0.35)
            self.act7 = ParametricTanh(alpha=1.4, beta=1.15, gamma=0.06)
            self.act8 = ParametricSigmoid(alpha=1.5, beta=1.2, gamma=-0.28)
            self.act9 = ParametricTanh(alpha=1.25, beta=1.05, gamma=0.04)
            self.act10 = ParametricSigmoid(alpha=1.4, beta=1.1, gamma=-0.22)
            self.act11 = ParametricTanh(alpha=1.1, beta=0.95, gamma=0.02)
            self.act12 = ParametricSigmoid(alpha=1.3, beta=1.0, gamma=-0.18)
        
        self.activation_outputs = {}  # 用于存储激活值
    
    def forward(self, x, return_embedding=False, record_activations=False):
        if self.use_parametric:
            # 前12層使用參數化激活
            x = self.dropout(self.act1(self.bn1(self.fc1(x))))
            if record_activations: self.activation_outputs['layer1'] = x.detach().cpu()
            x = self.dropout(self.act2(self.bn2(self.fc2(x))))
            if record_activations: self.activation_outputs['layer2'] = x.detach().cpu()
            x = self.dropout(self.act3(self.bn3(self.fc3(x))))
            if record_activations: self.activation_outputs['layer3'] = x.detach().cpu()
            x = self.dropout(self.act4(self.bn4(self.fc4(x))))
            if record_activations: self.activation_outputs['layer4'] = x.detach().cpu()
            x = self.dropout(self.act5(self.bn5(self.fc5(x))))
            if record_activations: self.activation_outputs['layer5'] = x.detach().cpu()
            x = self.dropout(self.act6(self.bn6(self.fc6(x))))
            if record_activations: self.activation_outputs['layer6'] = x.detach().cpu()
            x = self.dropout(self.act7(self.bn7(self.fc7(x))))
            if record_activations: self.activation_outputs['layer7'] = x.detach().cpu()
            x = self.dropout(self.act8(self.bn8(self.fc8(x))))
            if record_activations: self.activation_outputs['layer8'] = x.detach().cpu()
            x = self.dropout(self.act9(self.bn9(self.fc9(x))))
            if record_activations: self.activation_outputs['layer9'] = x.detach().cpu()
            x = self.dropout(self.act10(self.bn10(self.fc10(x))))
            if record_activations: self.activation_outputs['layer10'] = x.detach().cpu()
            x = self.dropout(self.act11(self.bn11(self.fc11(x))))
            if record_activations: self.activation_outputs['layer11'] = x.detach().cpu()
            x = self.dropout(self.act12(self.bn12(self.fc12(x))))
            if record_activations: self.activation_outputs['layer12'] = x.detach().cpu()
        else:
            # 前12層使用標準ReLU
            x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
            if record_activations: self.activation_outputs['layer1'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
            if record_activations: self.activation_outputs['layer2'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
            if record_activations: self.activation_outputs['layer3'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
            if record_activations: self.activation_outputs['layer4'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn5(self.fc5(x))))
            if record_activations: self.activation_outputs['layer5'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn6(self.fc6(x))))
            if record_activations: self.activation_outputs['layer6'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn7(self.fc7(x))))
            if record_activations: self.activation_outputs['layer7'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn8(self.fc8(x))))
            if record_activations: self.activation_outputs['layer8'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn9(self.fc9(x))))
            if record_activations: self.activation_outputs['layer9'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn10(self.fc10(x))))
            if record_activations: self.activation_outputs['layer10'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn11(self.fc11(x))))
            if record_activations: self.activation_outputs['layer11'] = x.detach().cpu()
            x = self.dropout(torch.relu(self.bn12(self.fc12(x))))
            if record_activations: self.activation_outputs['layer12'] = x.detach().cpu()
        
        # 後面8層都用標準ReLU（無論哪種模式）
        x = self.dropout(torch.relu(self.fc13(x)))
        x = self.dropout(torch.relu(self.fc14(x)))
        x = self.dropout(torch.relu(self.fc15(x)))
        x = self.dropout(torch.relu(self.fc16(x)))
        x = self.dropout(torch.relu(self.fc17(x)))
        x = self.dropout(torch.relu(self.fc18(x)))
        x = torch.relu(self.fc19(x))
        emb = x
        x = self.fc20(x)  # 移除 sigmoid（使用 BCEWithLogitsLoss）
        return (x, emb) if return_embedding else x


def calc_metrics(y_true, y_pred, y_prob):
    """計算分類指標"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'sens': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'spec': tn / (tn + fp) if (tn + fp) > 0 else 0
    }


def train_model(model, train_loader, val_loader, epochs=200, lr=0.001, use_scheduler=False):
    """訓練模型（混合精度 + 優化數據傳輸）"""
    criterion = nn.BCEWithLogitsLoss()  # 混合精度安全的損失函數
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()  # 混合精度梯度縮放器
    
    # 學習率調度器（餘弦退火）
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    early_stop = EarlyStopping(patience=30, start_delay=50, mode='min')
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_sens': [], 'train_spec': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_sens': [], 'val_spec': [],
        'lr': []  # 記錄學習率
    }
    
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        train_losses = []
        train_probs_list = []  # 改用列表累積，減少 CPU 傳輸
        train_labels_list = []
        
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度訓練（FP16）
            with autocast():
                out = model(X)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            train_probs_list.append(out.detach())
            train_labels_list.append(y)
        
        # 批量轉換到 CPU（減少傳輸次數）
        train_probs = torch.cat(train_probs_list)
        train_probs = torch.sigmoid(train_probs).cpu().numpy().flatten()  # 加 sigmoid
        train_labels = torch.cat(train_labels_list).cpu().numpy().flatten()
        train_preds = (train_probs >= 0.5).astype(int)
        train_metrics = calc_metrics(train_labels, train_preds, train_probs)
        
        # 驗證階段
        model.eval()
        val_losses = []
        val_probs_list = []
        val_labels_list = []
        
        with torch.no_grad(), autocast():  # 驗證也使用混合精度
            for X, y in val_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(X)
                loss = criterion(out, y)
                val_losses.append(loss.item())
                val_probs_list.append(out)
                val_labels_list.append(y)
        
        # 批量轉換到 CPU
        val_probs = torch.cat(val_probs_list)
        val_probs = torch.sigmoid(val_probs).cpu().numpy().flatten()  # 加 sigmoid
        val_labels = torch.cat(val_labels_list).cpu().numpy().flatten()
        val_preds = (val_probs >= 0.5).astype(int)
        val_metrics = calc_metrics(val_labels, val_preds, val_probs)
        
        # 記錄歷史
        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(train_metrics['acc'])
        history['train_auc'].append(train_metrics['auc'])
        history['train_sens'].append(train_metrics['sens'])
        history['train_spec'].append(train_metrics['spec'])
        
        history['val_loss'].append(np.mean(val_losses))
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_sens'].append(val_metrics['sens'])
        history['val_spec'].append(val_metrics['spec'])
        
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 調整學習率
        if scheduler:
            scheduler.step()
        
        # 改善日誌顯示：每 10 個 epoch 顯示詳細資訊
        if (epoch + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1:3d}/{epochs}] "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f} | "
                  f"Val AUC: {history['val_auc'][-1]:.4f} | "
                  f"Val Acc: {history['val_acc'][-1]:.4f} | "
                  f"LR: {history['lr'][-1]:.6f}")
        
        # 早停檢查
        if early_stop(history['val_loss'][-1], epoch):
            print(f"  ⚠️  Early Stopping at epoch {epoch+1} (最佳 epoch: {early_stop.best_epoch+1})")
            print(f"      最佳驗證 Loss: {early_stop.best_score:.4f}")
            break
    
    return history


def plot_fold_history(history, model_name, fold, save_path):
    """繪製單個fold的訓練歷程"""
    fig = plt.figure(figsize=(15, 10))
    metrics = [
        ('Loss', 'train_loss', 'val_loss'),
        ('Accuracy', 'train_acc', 'val_acc'),
        ('AUC', 'train_auc', 'val_auc'),
        ('Sensitivity', 'train_sens', 'val_sens'),
        ('Specificity', 'train_spec', 'val_spec'),
        ('Learning Rate', 'lr', None)
    ]
    
    for i, (title, train_key, val_key) in enumerate(metrics, 1):
        ax = plt.subplot(2, 3, i)
        epochs = range(1, len(history[train_key]) + 1)
        ax.plot(epochs, history[train_key], 'b-', label='Train', linewidth=2)
        if val_key:
            ax.plot(epochs, history[val_key], 'r-', label='Validation', linewidth=2)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)
    
    # 在右下角（第6個子圖位置）添加訓練摘要
    ax_summary = plt.subplot(2, 3, 6)
    ax_summary.axis('off')
    
    # 獲取最終的5個指標值
    total_epochs = len(history['train_loss'])
    summary_text = f"Training Summary\n\n"
    summary_text += f"Total Epochs: {total_epochs}\n\n"
    summary_text += "Final Validation Metrics:\n"
    summary_text += f"  Loss:        {history['val_loss'][-1]:.4f}\n"
    summary_text += f"  Accuracy:    {history['val_acc'][-1]:.4f}\n"
    summary_text += f"  AUC:         {history['val_auc'][-1]:.4f}\n"
    summary_text += f"  Sensitivity: {history['val_sens'][-1]:.4f}\n"
    summary_text += f"  Specificity: {history['val_spec'][-1]:.4f}\n\n"
    summary_text += f"Best Validation Metrics:\n"
    summary_text += f"  Best AUC:    {max(history['val_auc']):.4f}\n"
    summary_text += f"  Best Acc:    {max(history['val_acc']):.4f}\n"
    
    ax_summary.text(0.1, 0.95, summary_text, 
                   transform=ax_summary.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{model_name} - Fold {fold} Training History', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已儲存: {save_path}")


def save_statistics_csv(all_histories, model_name, save_path):
    """將所有fold的統計資料儲存為CSV"""
    # 找出最短的訓練長度
    min_len = min(len(h['train_loss']) for h in all_histories)
    
    # 收集所有fold的數據
    metrics = ['train_loss', 'train_acc', 'train_auc', 'train_sens', 'train_spec',
               'val_loss', 'val_acc', 'val_auc', 'val_sens', 'val_spec']
    
    stats_data = []
    for epoch in range(min_len):
        row = {'Epoch': epoch + 1, 'Model': model_name}
        
        for metric in metrics:
            values = [h[metric][epoch] for h in all_histories]
            row[f'{metric}_mean'] = np.mean(values)
            row[f'{metric}_std'] = np.std(values)
            row[f'{metric}_min'] = np.min(values)
            row[f'{metric}_max'] = np.max(values)
        
        stats_data.append(row)
    
    df = pd.DataFrame(stats_data)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"  已儲存統計CSV: {save_path}")


def plot_average_history(all_histories, model_name, save_path):
    """繪製平均訓練歷程（含標準差陰影）"""
    min_len = min(len(h['train_loss']) for h in all_histories)
    
    fig = plt.figure(figsize=(15, 10))
    metrics = [
        ('Loss', 'train_loss', 'val_loss'),
        ('Accuracy', 'train_acc', 'val_acc'),
        ('AUC', 'train_auc', 'val_auc'),
        ('Sensitivity', 'train_sens', 'val_sens'),
        ('Specificity', 'train_spec', 'val_spec')
    ]
    
    for i, (title, train_key, val_key) in enumerate(metrics, 1):
        ax = plt.subplot(2, 3, i)
        epochs = range(1, min_len + 1)
        
        # 訓練數據
        train_data = np.array([h[train_key][:min_len] for h in all_histories])
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        ax.plot(epochs, train_mean, 'b-', label='Train (mean)', linewidth=2)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue', label='Train (±1 std)')
        
        # 驗證數據
        val_data = np.array([h[val_key][:min_len] for h in all_histories])
        val_mean = np.mean(val_data, axis=0)
        val_std = np.std(val_data, axis=0)
        ax.plot(epochs, val_mean, 'r-', label='Validation (mean)', linewidth=2)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='red', label='Validation (±1 std)')
        
        ax.set_title(f'{title} (Mean ± Std)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # 在右下角（第6個子圖位置）添加平均訓練摘要
    ax_summary = plt.subplot(2, 3, 6)
    ax_summary.axis('off')
    
    # 計算最終的平均指標
    summary_text = f"Average Training Summary\n\n"
    summary_text += f"Average Epochs: {min_len}\n"
    summary_text += f"Folds: {len(all_histories)}\n\n"
    summary_text += "Final Val Metrics (mean±std):\n"
    
    for metric_name, key in [('Loss', 'val_loss'), ('Acc', 'val_acc'), ('AUC', 'val_auc'), 
                              ('Sens', 'val_sens'), ('Spec', 'val_spec')]:
        final_values = [h[key][-1] for h in all_histories]
        mean_val = np.mean(final_values)
        std_val = np.std(final_values)
        summary_text += f"  {metric_name:5s}: {mean_val:.4f}±{std_val:.4f}\n"
    
    # 最佳AUC
    best_aucs = [max(h['val_auc']) for h in all_histories]
    summary_text += f"\nBest AUC (mean): {np.mean(best_aucs):.4f}\n"
    
    ax_summary.text(0.1, 0.95, summary_text, 
                   transform=ax_summary.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'{model_name} - Average Training History (5 Folds)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已儲存平均訓練圖: {save_path}")


def plot_model_comparison(results, save_path='results/model_comparison.png'):
    """繪製3個模型的性能比較圖 - 學術發表級"""
    print("\n繪製模型性能比較圖...")
    
    # 使用白色背景，去掉網格
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    model_names = list(results.keys())
    colors = ['#87CEEB', '#FFB6C1', '#90EE90']  # 淺藍、淺紅、淺綠
    
    # ==================== 左圖：平均 AUC 柱狀圖 ====================
    ax1 = axes[0]
    ax1.set_facecolor('white')
    
    mean_aucs = [results[name]['mean_auc'] for name in model_names]
    std_aucs = [results[name]['std'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    bars1 = ax1.bar(x_pos, mean_aucs, yerr=std_aucs, capsize=8, 
                    color=colors, alpha=0.9, edgecolor='black', linewidth=2,
                    error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    ax1.set_xlabel('Model Architecture', fontsize=13, fontweight='bold')
    ax1.set_ylabel('AUC Score', fontsize=13, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, fontsize=12, fontweight='bold')
    ax1.set_ylim([0.8, 1.0])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.tick_params(width=1.5, labelsize=11)
    
    # 在柱子上方標註數值（粗體）
    for i, (bar, mean, std) in enumerate(zip(bars1, mean_aucs, std_aucs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.012,
                f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 標記最佳模型 - 黃底黑字標籤
    best_idx = mean_aucs.index(max(mean_aucs))
    ax1.text(best_idx, 0.985, '🏆 Best',
            ha='center', va='center', fontsize=12, fontweight='bold', 
            color='black', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', 
                     edgecolor='black', linewidth=2, alpha=0.9))
    
    # ==================== 右圖：5-Fold 分佈箱線圖 ====================
    ax2 = axes[1]
    ax2.set_facecolor('white')
    
    positions = []
    all_fold_aucs = []
    for i, name in enumerate(model_names):
        fold_aucs = results[name]['fold_aucs']
        all_fold_aucs.append(fold_aucs)
        positions.append(i)
    
    # 繪製箱線圖
    bp = ax2.boxplot(all_fold_aucs, positions=positions, widths=0.5,
                     patch_artist=True, showmeans=False,
                     medianprops=dict(color='black', linewidth=2.5, label='Median'),
                     boxprops=dict(edgecolor='black', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     flierprops=dict(marker='o', markerfacecolor='red', markersize=6, 
                                   markeredgecolor='black'))
    
    # 為每個箱子設置顏色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    
    # 手動添加平均值標記（紅色鑽石）
    means = [np.mean(aucs) for aucs in all_fold_aucs]
    ax2.plot(positions, means, marker='D', color='red', markersize=10, 
            linestyle='', markeredgecolor='black', markeredgewidth=1.5,
            label='Mean', zorder=5)
    
    ax2.set_xlabel('Model Architecture', fontsize=13, fontweight='bold')
    ax2.set_ylabel('AUC Score', fontsize=13, fontweight='bold')
    ax2.set_title('AUC Distribution Across 5 Folds', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(model_names, fontsize=12, fontweight='bold')
    ax2.set_ylim([0.8, 1.0])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.tick_params(width=1.5, labelsize=11)
    
    # 添加圖例（只顯示Median和Mean）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2.5, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
              markeredgecolor='black', markeredgewidth=1.5, markersize=10, 
              linestyle='', label='Mean')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=11, 
              frameon=True, fancybox=False, edgecolor='black', framealpha=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已儲存模型比較圖: {save_path}")


def experiment_a(X_dev, y_dev):
    """實驗 A：模型架構比較（改進版：輸出所有fold的圖表和統計）"""
    print("="*70)
    print("實驗 A：模型架構比較（進階版）")
    print("="*70)
    
    models = {
        'Shallow': ShallowMLP,
        'Medium': MediumMLP,
        'Deep': DeepMLP
    }
    
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for name, model_cls in models.items():
        print(f"\n{'='*70}")
        print(f"訓練 {name} 模型（5-Fold 交叉驗證）")
        print(f"{'='*70}")
        fold_aucs = []
        fold_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), 1):
            print(f"\n📊 Fold {fold}/5 - 訓練樣本: {len(train_idx)}, 驗證樣本: {len(val_idx)}")
            X_tr, X_val = X_dev[train_idx], X_dev[val_idx]
            y_tr, y_val = y_dev[train_idx], y_dev[val_idx]
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            
            # 優化 DataLoader：大 batch + 多 worker + pin_memory（針對 i9-12900 + RTX A6000）
            train_loader = DataLoader(
                MLP_Dataset(X_tr, y_tr), 
                batch_size=512,           # 從 64 提升到 512（8倍）
                shuffle=True,
                num_workers=12,           # 使用 12 個 CPU 核心（16核的75%）
                pin_memory=True,          # 加速 CPU→GPU 傳輸
                persistent_workers=True,  # 避免 worker 重啟開銷
                prefetch_factor=4,        # 預載 4 批數據
                drop_last=True            # 確保批次大小一致
            )
            val_loader = DataLoader(
                MLP_Dataset(X_val, y_val), 
                batch_size=1024,          # 驗證可用更大 batch（無梯度）
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
            
            model = model_cls().to(device)
            history = train_model(model, train_loader, val_loader, epochs=200)
            
            # 不儲存每個fold的單獨訓練圖，只保留歷史記錄用於計算平均值
            best_auc = max(history['val_auc'])
            best_epoch = history['val_auc'].index(best_auc) + 1
            final_auc = history['val_auc'][-1]
            
            fold_aucs.append(best_auc)
            fold_histories.append(history)
            print(f"\n  ✅ Fold {fold} 完成:")
            print(f"     最佳 AUC: {best_auc:.4f} (Epoch {best_epoch})")
            print(f"     最終 AUC: {final_auc:.4f} (Epoch {len(history['val_auc'])})")
            print(f"     訓練 Epochs: {len(history['train_loss'])}")
        
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        results[name] = {'mean_auc': mean_auc, 'std': std_auc, 'fold_aucs': fold_aucs}
        
        print(f"\n{name} 平均驗證 AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  各Fold AUC: {', '.join([f'{auc:.4f}' for auc in fold_aucs])}")
        
        # 儲存統計CSV
        stats_csv_path = f'results/statistics/{name}_training_statistics.csv'
        save_statistics_csv(fold_histories, name, stats_csv_path)
        
        # 繪製平均訓練圖（含標準差）
        avg_plot_path = f'results/{name}_history_avg.png'
        plot_average_history(fold_histories, name, avg_plot_path)
    
    best_model = max(results, key=lambda k: results[k]['mean_auc'])
    print(f"\n{'='*70}")
    print(f"最佳模型: {best_model} (AUC={results[best_model]['mean_auc']:.4f})")
    print(f"{'='*70}\n")
    
    # 繪製模型比較圖
    plot_model_comparison(results)
    
    return results, best_model


def analyze_feature_importance(model, X_test, y_test, feature_names, save_path):
    """分析特徵重要性（使用梯度方法）"""
    print("\n分析特徵重要性...")
    
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device, non_blocking=True)
    X_tensor.requires_grad = True
    
    with autocast():
        output = model(X_tensor)
    
    loss = nn.BCEWithLogitsLoss()(output, torch.FloatTensor(y_test).unsqueeze(1).to(device, non_blocking=True))
    loss.backward()
    
    # 計算每個特徵的平均絕對梯度
    gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    
    # 排序找出最重要的20個特徵
    top_indices = np.argsort(gradients)[-20:][::-1]
    top_importance = gradients[top_indices]
    top_features = [f"Feature_{i}" for i in top_indices]
    
    # 繪製特徵重要性
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_importance, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Average Absolute Gradient', fontsize=12)
    ax.set_title('Top 20 Feature Importance (Gradient-based)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # 添加數值標籤
    for i, (bar, val) in enumerate(zip(bars, top_importance)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已儲存特徵重要性圖: {save_path}")
    
    return top_indices, top_importance


def plot_decision_boundary_pca(model, X_test, y_test, save_path):
    """使用PCA降維並繪製決策邊界"""
    print("\n繪製決策邊界（PCA降維）...")
    
    # PCA降至2D
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_test)
    
    # 創建網格
    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 反向轉換網格點到原始空間
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_pca)
    
    # 預測
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(grid_original).to(device)
        Z = model(grid_tensor).cpu().numpy()
    Z = Z.reshape(xx.shape)
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.7)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm', 
                        edgecolors='k', s=50, alpha=0.8)
    
    # 決策邊界
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    plt.colorbar(contour, ax=ax, label='Predicted Probability')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Decision Boundary (PCA 2D Projection)', fontsize=14, fontweight='bold')
    ax.legend(*scatter.legend_elements(), title='True Label', loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已儲存決策邊界圖: {save_path}")


def plot_calibration_curve(y_true, y_prob, save_path):
    """繪製校準曲線（Calibration Curve）"""
    print("\n繪製校準曲線...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左圖：校準曲線
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy='uniform'
    )
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    ax1.plot(mean_predicted_value, fraction_of_positives, 'o-', 
            label='Model', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 右圖：預測概率分布直方圖
    ax2.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label='Benign (0)', 
            color='#3498db', edgecolor='black')
    ax2.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label='Malignant (1)', 
            color='#e74c3c', edgecolor='black')
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已儲存校準曲線圖: {save_path}")


def analyze_misclassified_samples(X_test, y_test, y_pred, y_prob, save_path):
    """分析誤分類樣本"""
    print("\n分析誤分類樣本...")
    
    # 找出誤分類樣本
    misclassified = y_test != y_pred
    fp_mask = (y_test == 0) & (y_pred == 1)  # False Positive
    fn_mask = (y_test == 1) & (y_pred == 0)  # False Negative
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 誤分類樣本的預測概率分布
    ax = axes[0, 0]
    fp_probs = y_prob[fp_mask]
    fn_probs = y_prob[fn_mask]
    
    ax.hist(fp_probs, bins=20, alpha=0.6, label=f'False Positive (n={len(fp_probs)})', 
           color='orange', edgecolor='black')
    ax.hist(fn_probs, bins=20, alpha=0.6, label=f'False Negative (n={len(fn_probs)})', 
           color='purple', edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Misclassified Samples - Probability Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 2. 預測信心分析
    ax = axes[0, 1]
    correct_conf = np.abs(y_prob[~misclassified] - 0.5)
    incorrect_conf = np.abs(y_prob[misclassified] - 0.5)
    
    ax.boxplot([correct_conf, incorrect_conf], labels=['Correct', 'Incorrect'], 
               patch_artist=True, 
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax.set_ylabel('Prediction Confidence |p - 0.5|', fontsize=12)
    ax.set_title('Prediction Confidence Comparison', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # 3. 誤分類樣本數量統計
    ax = axes[1, 0]
    categories = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
    tp = ((y_test == 1) & (y_pred == 1)).sum()
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = fp_mask.sum()
    fn = fn_mask.sum()
    counts = [tp, tn, fp, fn]
    colors = ['#27ae60', '#3498db', '#e67e22', '#e74c3c']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Result Breakdown', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # 添加數值標籤
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 4. 錯誤率分析
    ax = axes[1, 1]
    total = len(y_test)
    error_rate = misclassified.sum() / total * 100
    fp_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    
    metrics_names = ['Overall\nError Rate', 'False Positive\nRate', 'False Negative\nRate']
    metrics_values = [error_rate, fp_rate, fn_rate]
    colors_bar = ['#95a5a6', '#e67e22', '#e74c3c']
    
    bars = ax.bar(metrics_names, metrics_values, color=colors_bar, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Error Rate (%)', fontsize=12)
    ax.set_title('Error Rate Analysis', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, max(metrics_values) * 1.2])
    
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已儲存誤分類分析圖: {save_path}")


def get_parametric_layers_count(model_name):
    """獲取不同模型的參數化層數"""
    layers_map = {
        'Shallow': 3,  # 5層模型，前3層參數化
        'Medium': 6,   # 10層模型，前6層參數化
        'Deep': 12     # 20層模型，前12層參數化
    }
    return layers_map.get(model_name, 12)


def plot_confusion_matrices(y_true, y_pred_relu, y_pred_param, save_path, model_name='Deep'):
    """繪製混淆矩陣對比圖（ReLU vs Parametric）"""
    print("\n繪製混淆矩陣對比...")
    
    cm_relu = confusion_matrix(y_true, y_pred_relu)
    cm_param = confusion_matrix(y_true, y_pred_param)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Confusion Matrix Comparison ({model_name} Model)', fontsize=16, fontweight='bold')
    
    # ReLU混淆矩陣
    ax1 = axes[0]
    sns.heatmap(cm_relu, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_title('ReLU Activation', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # 計算ReLU指標
    tn_r, fp_r, fn_r, tp_r = cm_relu.ravel()
    acc_r = (tp_r + tn_r) / (tp_r + tn_r + fp_r + fn_r)
    sens_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0
    spec_r = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0
    
    text_r = f'Accuracy: {acc_r:.4f}\nSensitivity: {sens_r:.4f}\nSpecificity: {spec_r:.4f}'
    ax1.text(0.5, -0.15, text_r, transform=ax1.transAxes, 
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Parametric混淆矩陣
    ax2 = axes[1]
    sns.heatmap(cm_param, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'}, ax=ax2)
    ax2.set_title('Parametric Tanh/Sigmoid', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    # 計算Parametric指標
    tn_p, fp_p, fn_p, tp_p = cm_param.ravel()
    acc_p = (tp_p + tn_p) / (tp_p + tn_p + fp_p + fn_p)
    sens_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0
    spec_p = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else 0
    
    text_p = f'Accuracy: {acc_p:.4f}\nSensitivity: {sens_p:.4f}\nSpecificity: {spec_p:.4f}'
    ax2.text(0.5, -0.15, text_p, transform=ax2.transAxes,
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存混淆矩陣: {save_path}")


def plot_5metrics_comparison(history_relu, history_param, save_path, model_name='Deep'):
    """繪製5個指標的訓練曲線對比圖（Loss, Accuracy, AUC, Sensitivity, Specificity）"""
    print("\n繪製5指標訓練曲線對比...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics Comparison ({model_name} Model): ReLU vs Parametric Activation', 
                 fontsize=16, fontweight='bold')
    
    metrics = [
        ('Loss', 'val_loss', 'lower'),
        ('Accuracy', 'val_acc', 'upper'),
        ('AUC', 'val_auc', 'upper'),
        ('Sensitivity', 'val_sens', 'upper'),
        ('Specificity', 'val_spec', 'upper')
    ]
    
    for idx, (title, key, better) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        epochs_relu = range(1, len(history_relu[key]) + 1)
        epochs_param = range(1, len(history_param[key]) + 1)
        
        ax.plot(epochs_relu, history_relu[key], linewidth=2.5, 
                color='#3498db', label='ReLU', alpha=0.8)
        ax.plot(epochs_param, history_param[key], linewidth=2.5, 
                color='#e74c3c', label='Parametric', alpha=0.8)
        
        # 標記最佳值
        if better == 'lower':
            best_relu = min(history_relu[key])
            best_param = min(history_param[key])
        else:
            best_relu = max(history_relu[key])
            best_param = max(history_param[key])
        
        ax.axhline(y=best_relu, color='#3498db', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=best_param, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        # 添加最佳值文本
        final_relu = history_relu[key][-1]
        final_param = history_param[key][-1]
        textstr = f'Final ReLU: {final_relu:.4f}\nFinal Param: {final_param:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 第6個子圖：參數比較摘要
    ax_summary = axes[1, 2]
    ax_summary.axis('off')
    
    summary_text = "Performance Summary\n\n"
    summary_text += "Final Validation Metrics:\n\n"
    
    for title, key, better in metrics:
        val_relu = history_relu[key][-1]
        val_param = history_param[key][-1]
        diff = val_param - val_relu
        symbol = '↑' if diff > 0 else '↓'
        
        if (better == 'upper' and diff > 0) or (better == 'lower' and diff < 0):
            color_code = '\033[92m'  # 綠色表示改善
        else:
            color_code = '\033[91m'  # 紅色表示退化
        
        summary_text += f"{title}:\n"
        summary_text += f"  ReLU: {val_relu:.4f}\n"
        summary_text += f"  Param: {val_param:.4f} ({symbol}{abs(diff):.4f})\n\n"
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存5指標對比圖: {save_path}")


def experiment_b_advanced(X_dev, y_dev, X_test, y_test, best_model_name):
    """實驗 B：激活函數對比實驗（ReLU vs Parametric Tanh/Sigmoid）"""
    print("="*70)
    print(f"實驗 B：激活函數對比與參數學習分析 (基於 {best_model_name} 模型)")
    print("="*70)
    
    models_map = {'Shallow': ShallowMLP, 'Medium': MediumMLP, 'Deep': DeepMLP}
    model_cls = models_map[best_model_name]
    
    # 獲取該模型的參數化層數
    param_layers = get_parametric_layers_count(best_model_name)
    print(f"\n📊 使用 {best_model_name} 模型進行對比實驗")
    print(f"   - 前 {param_layers} 層將使用參數化 Tanh/Sigmoid 激活函數")
    print(f"   - 其餘層使用標準 ReLU 激活函數")
    
    use_parametric = True  # 一定進行參數化對比
    
    scaler = StandardScaler()
    X_dev_sc = scaler.fit_transform(X_dev)
    X_test_sc = scaler.transform(X_test)
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_dev_sc, y_dev, test_size=0.1, 
                                                  random_state=RANDOM_SEED, stratify=y_dev)
    
    # 優化 DataLoader（高性能配置）
    train_loader = DataLoader(
        MLP_Dataset(X_tr, y_tr), 
        batch_size=512, 
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )
    val_loader = DataLoader(
        MLP_Dataset(X_val, y_val), 
        batch_size=1024,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        MLP_Dataset(X_test_sc, y_test), 
        batch_size=1024,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    results_comparison = {}
    
    if use_parametric:
        # ========== 1. 訓練標準ReLU版本（Baseline） ==========
        print("\n" + "="*70)
        print(f"階段 1/2: 訓練標準 ReLU 激活函數版本（Baseline - {best_model_name}）")
        print("="*70)
        
        model_relu = model_cls(use_parametric_activation=False).to(device)
        history_relu = train_model(model_relu, train_loader, val_loader, epochs=200, use_scheduler=True)
        
        # 評估ReLU版本
        model_relu.eval()
        test_probs_relu_list = []
        test_labels_relu_list = []
        with torch.no_grad(), autocast():
            for X, y in test_loader:
                X = X.to(device, non_blocking=True)
                out, _ = model_relu(X, return_embedding=True)
                test_probs_relu_list.append(out)
                test_labels_relu_list.append(y)
        
        test_probs_relu = torch.cat(test_probs_relu_list)
        test_probs_relu = torch.sigmoid(test_probs_relu).cpu().numpy().flatten()  # 加 sigmoid
        test_labels_relu = torch.cat(test_labels_relu_list).numpy().flatten()
        test_preds_relu = (test_probs_relu >= 0.5).astype(int)
        metrics_relu = calc_metrics(test_labels_relu, test_preds_relu, test_probs_relu)
        
        results_comparison['ReLU'] = {
            'model': model_relu,
            'history': history_relu,
            'metrics': metrics_relu,
            'test_probs': test_probs_relu,
            'test_preds': test_preds_relu
        }
        
        print(f"\n✅ ReLU版本測試集結果:")
        print(f"   AUC: {metrics_relu['auc']:.4f}, Acc: {metrics_relu['acc']:.4f}")
        
        # ========== 2. 訓練參數化Tanh/Sigmoid版本 ==========
        print("\n" + "="*70)
        print(f"階段 2/2: 訓練參數化 Tanh/Sigmoid 激活函數版本 ({best_model_name})")
        print("="*70)
        
        model_param = model_cls(use_parametric_activation=True).to(device)
        
        # 記錄初始參數
        initial_params = {}
        print("\n📊 初始激活函數參數:")
        for name, module in model_param.named_modules():
            if isinstance(module, (ParametricTanh, ParametricSigmoid)):
                act_type = 'Tanh' if isinstance(module, ParametricTanh) else 'Sigmoid'
                alpha_init = module.alpha.item()
                beta_init = module.beta.item()
                gamma_init = module.gamma.item()
                initial_params[name] = {
                    'type': act_type,
                    'alpha': alpha_init,
                    'beta': beta_init,
                    'gamma': gamma_init
                }
                print(f"  {name} ({act_type}): α={alpha_init:.3f}, β={beta_init:.3f}, γ={gamma_init:.3f}")
        
        history_param = train_model(model_param, train_loader, val_loader, epochs=200, use_scheduler=True)
        
        # 記錄訓練後參數
        final_params = {}
        print("\n📊 訓練後激活函數參數:")
        for name, module in model_param.named_modules():
            if isinstance(module, (ParametricTanh, ParametricSigmoid)):
                act_type = 'Tanh' if isinstance(module, ParametricTanh) else 'Sigmoid'
                alpha_final = module.alpha.item()
                beta_final = module.beta.item()
                gamma_final = module.gamma.item()
                final_params[name] = {
                    'type': act_type,
                    'alpha': alpha_final,
                    'beta': beta_final,
                    'gamma': gamma_final
                }
                
                # 計算變化量
                alpha_change = alpha_final - initial_params[name]['alpha']
                beta_change = beta_final - initial_params[name]['beta']
                gamma_change = gamma_final - initial_params[name]['gamma']
                
                print(f"  {name} ({act_type}): α={alpha_final:.3f} (Δ{alpha_change:+.3f}), "
                      f"β={beta_final:.3f} (Δ{beta_change:+.3f}), γ={gamma_final:.3f} (Δ{gamma_change:+.3f})")
        
        # 評估參數化版本
        model_param.eval()
        test_probs_param_list = []
        test_labels_param_list = []
        with torch.no_grad(), autocast():
            for X, y in test_loader:
                X = X.to(device, non_blocking=True)
                out, _ = model_param(X, return_embedding=True)
                test_probs_param_list.append(out)
                test_labels_param_list.append(y)
        
        test_probs_param = torch.cat(test_probs_param_list)
        test_probs_param = torch.sigmoid(test_probs_param).cpu().numpy().flatten()  # 加 sigmoid
        test_labels_param = torch.cat(test_labels_param_list).numpy().flatten()
        test_preds_param = (test_probs_param >= 0.5).astype(int)
        metrics_param = calc_metrics(test_labels_param, test_preds_param, test_probs_param)
        
        results_comparison['Parametric'] = {
            'model': model_param,
            'history': history_param,
            'metrics': metrics_param,
            'test_probs': test_probs_param,
            'test_preds': test_preds_param,
            'initial_params': initial_params,
            'final_params': final_params
        }
        
        print(f"\n✅ Parametric版本測試集結果:")
        print(f"   AUC: {metrics_param['auc']:.4f}, Acc: {metrics_param['acc']:.4f}")
        print(f"\n📈 性能提升: AUC改善 {(metrics_param['auc'] - metrics_relu['auc'])*100:.2f}%")
        
        # ========== 繪製混淆矩陣 ==========
        plot_confusion_matrices(test_labels_param, test_preds_relu, test_preds_param,
                              f'results/advanced/confusion_matrices_{best_model_name}.png',
                              model_name=best_model_name)
        
        # ========== 繪製5指標訓練曲線對比 ==========
        plot_5metrics_comparison(history_relu, history_param,
                               f'results/advanced/5metrics_training_{best_model_name}.png',
                               model_name=best_model_name)
        
    else:
        # 非Deep模型，使用標準流程
        model = model_cls().to(device)
        history = train_model(model, train_loader, val_loader, epochs=200, use_scheduler=True)
        
        model.eval()
        test_probs_list = []
        test_labels_list = []
        with torch.no_grad(), autocast():
            for X, y in test_loader:
                X = X.to(device, non_blocking=True)
                out, _ = model(X, return_embedding=True)
                test_probs_list.append(out)
                test_labels_list.append(y)
        
        test_probs = torch.cat(test_probs_list)
        test_probs = torch.sigmoid(test_probs).cpu().numpy().flatten()  # 加 sigmoid
        test_labels = torch.cat(test_labels_list).numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(int)
        metrics = calc_metrics(test_labels, test_preds, test_probs)
    
    # ========== 3. 生成對比分析圖表 ==========
    print("\n" + "="*70)
    print("生成對比分析與可視化")
    print("="*70)
    
    if use_parametric:
        # 繪製學習率曲線
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history_relu['lr'], linewidth=2, color='#3498db', label='ReLU', alpha=0.7)
        ax.plot(history_param['lr'], linewidth=2, color='#e74c3c', label='Parametric', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('results/advanced/learning_rate_schedule.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ 已儲存學習率曲線")
        
        # 生成完整對比分析
        from activation_utils import (
            plot_activation_comparison,
            plot_parameter_changes,
            plot_activation_function_shapes,
            plot_performance_comparison,
            save_parameter_comparison_table
        )
        
        # 1. 激活函數參數變化圖
        plot_parameter_changes(initial_params, final_params, 
                              'results/advanced/parameter_changes.png')
        
        # 2. 激活函數形狀對比
        plot_activation_function_shapes(initial_params, final_params,
                                       'results/advanced/activation_shapes.png')
        
        # 3. 性能對比圖
        plot_performance_comparison(results_comparison,
                                   'results/advanced/performance_comparison.png')
        
        # 4. 激活值分布對比
        plot_activation_comparison(model_relu, model_param, test_loader, device,
                                  'results/advanced/activation_distribution_comparison.png')
        
        # 5. 保存參數對比表
        save_parameter_comparison_table(initial_params, final_params,
                                       'results/advanced/parameter_comparison.csv')
        
        # 使用參數化模型進行其他分析
        model = model_param
        test_labels = test_labels_param
        test_preds = test_preds_param
        test_probs = test_probs_param
        metrics = metrics_param
    
    # 繼續其他標準分析
    print("\n執行標準進階分析...")
    
    # 特徵重要性分析
    analyze_feature_importance(model, X_test_sc, y_test, None, 
                              'results/advanced/feature_importance.png')
    
    # 決策邊界可視化
    plot_decision_boundary_pca(model, X_test_sc, y_test, 
                              'results/advanced/decision_boundary.png')
    
    # 校準曲線
    plot_calibration_curve(test_labels, test_probs, 
                          'results/advanced/calibration_curve.png')
    
    # 誤分類樣本分析
    analyze_misclassified_samples(X_test_sc, test_labels, test_preds, test_probs,
                                  'results/advanced/misclassification_analysis.png')
    
    print("\n" + "="*70)
    print("✅ 所有進階分析完成！")
    print("="*70)
    
    if use_parametric:
        print(f"\n📊 最終結果總結:")
        print(f"  ReLU版本      - AUC: {metrics_relu['auc']:.4f}, Acc: {metrics_relu['acc']:.4f}")
        print(f"  Parametric版本 - AUC: {metrics_param['auc']:.4f}, Acc: {metrics_param['acc']:.4f}")
        print(f"  性能提升: {(metrics_param['auc'] - metrics_relu['auc'])*100:+.2f}%")
    
    return metrics


def main():
    """主程式"""
    # 輸出裝置資訊（只在主進程）
    print(f"\n使用裝置: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA 優化: cudnn.benchmark={torch.backends.cudnn.benchmark}, TF32={torch.backends.cuda.matmul.allow_tf32}")
    print()
    
    print("="*70)
    print("載入資料...")
    print("="*70)
    
    df = pd.read_csv('features_160.csv')
    # 排除id列和label列
    feature_cols = [col for col in df.columns if col not in ['id', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"資料形狀: {X.shape}")
    print(f"類別分布: Benign(0)={np.sum(y==0)}, Malignant(1)={np.sum(y==1)}\n")
    
    # 切分資料
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"開發集: {X_dev.shape[0]} 樣本")
    print(f"測試集: {X_test.shape[0]} 樣本\n")
    
    # 實驗 A
    exp_a_results, best_model = experiment_a(X_dev, y_dev)
    
    # 實驗 B
    exp_b_metrics = experiment_b_advanced(X_dev, y_dev, X_test, y_test, best_model)
    
    print("\n" + "="*70)
    print("🎉 所有實驗完成！")
    print("="*70)
    print(f"\n✅ 已生成的圖表和分析：")
    print(f"  📁 results/folds/ - 每個fold的訓練圖 (9張)")
    print(f"  📁 results/statistics/ - 訓練統計CSV (3個)")
    print(f"  📁 results/ - 平均訓練圖 (3張)")
    print(f"  📁 results/advanced/ - 進階分析圖 (7張+)")
    print(f"    • 混淆矩陣對比圖 (新增)")
    print(f"    • 5指標訓練曲線對比 (新增)")
    print(f"    • 學習率調度曲線")
    print(f"    • 激活函數參數變化")
    print(f"    • 激活函數形狀對比")
    print(f"    • 性能對比圖")
    print(f"    • 激活值分布對比")
    print(f"    • 特徵重要性分析")
    print(f"    • 決策邊界可視化")
    print(f"    • 校準曲線")
    print(f"    • 誤分類樣本分析")
    print(f"\n🏆 最佳模型: {best_model}")
    print(f"📈 測試集 AUC: {exp_b_metrics['auc']:.4f}")


if __name__ == '__main__':
    main()
