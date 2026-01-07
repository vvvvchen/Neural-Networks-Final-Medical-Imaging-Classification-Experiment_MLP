# 類神經網路期末醫學影像分類實驗 - MLP

## 📋 專案資訊

| 項目 | 說明 |
|------|------|
| **主要執行檔** | `mlp_advanced.py` |
| **分類標籤檔** | `Classification_ok_labels.csv`（1萬筆數據，各5千正負標籤）|
| **特徵檔案** | `features_160.csv`（提取的160個特徵）|
| **資料集來源** | [Kaggle - Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/data) |

---

## 📊 專案概述

本專案實作了三種不同深度的 MLP 模型進行**醫學影像二元分類**（癌症檢測），並支援**參數化激活函數**以提升模型表達能力。

### 三種模型架構

| 模型 | 總層數 | 架構 | Dropout | 參數化激活層 |
|------|--------|------|---------|--------------|
| **Shallow** | 5層 | 160→128→96→64→32→1 | 0.2 | 前3層 |
| **Medium** | 10層 | 160→256→224→...→32→1 | 0.3 | 前6層 |
| **Deep** | 20層 | 160→384→352→...→8→1 | 0.5 | 前12層 |

---

## 📁 檔案結構總覽

### 🔹 核心程式檔案

| 檔案 | 說明 |
|------|------|
| `mlp_advanced.py` | **主程式** - MLP二元分類進階版，包含三種模型架構、5-Fold交叉驗證、混合精度訓練、參數化激活函數等完整實作 |
| `activation_utils.py` | 激活函數分析工具，提供參數變化視覺化與對比分析功能 |

### 🔹 資料檔案

| 檔案 | 說明 |
|------|------|
| `features_160.csv` | 從醫學影像提取的160維特徵向量 |
| `Classification_ok_labels.csv` | 分類好的1萬筆標籤（正負各5千筆）|
| `Classification_ok_labels/` | 標籤資料備份目錄 |

### 🔹 結果查看工具

| 檔案 | 說明 |
|------|------|
| `查看結果.py` | 快速開啟結果報告，顯示最佳模型性能摘要 |
| `查看進階結果.py` | 列出所有進階分析輸出，包含圖表與統計 CSV |
| `查看右下角數值.py` | 查看訓練圖表中的數值摘要與使用提示 |

### 🔹 說明文件

| 檔案 | 說明 |
|------|------|
| `模型架構詳細說明.md` | 三種模型（Shallow/Medium/Deep）的完整架構圖與參數說明 |
| `参數化激活函数說明.md` | ParametricTanh 與 ParametricSigmoid 的公式、設計原理與配置 |
| `參數調整詳細說明.txt` | 激活函數參數調整總覽，包含各層變化量與百分比 |
| `参數調整速查表.txt` | 激活函數參數的快速查閱表與設計原則 |
| `activation_parameters_config.md` | 激活函數參數配置檔，對應程式碼中的設定 |
| `專案完成總結.md` | 專案完成項目清單與實驗結果總結 |

### 🔹 其他檔案

| 檔案 | 說明 |
|------|------|
| `activation_parameters_changes.csv` | 激活函數參數變化的 CSV 記錄 |
| `prompt.txt` | 開發過程中的提示記錄 |

---

## 📁 輸出結果目錄

### 目錄結構：`results/`

```
results/
├── Shallow_history_avg.png    # 淺層模型平均訓練曲線（5-Fold, 含標準差）
├── Medium_history_avg.png     # 中層模型平均訓練曲線（5-Fold, 含標準差）
├── Deep_history_avg.png       # 深層模型平均訓練曲線（5-Fold, 含標準差）
├── model_comparison.png       # 三種模型 AUC 性能比較圖
├── folds/                     # 每個 fold 的詳細訓練記錄
├── statistics/                # 訓練統計 CSV 檔案
│   ├── Shallow_training_statistics.csv
│   ├── Medium_training_statistics.csv
│   └── Deep_training_statistics.csv
└── advanced/                  # 進階分析圖表
    ├── 5metrics_training_Medium.png      # 5指標訓練曲線
    ├── confusion_matrices_Medium.png     # 混淆矩陣對比
    ├── parameter_changes.png             # 激活函數參數變化
    ├── activation_shapes.png             # 激活函數形狀對比
    ├── activation_distribution_comparison.png  # 激活值分布
    ├── performance_comparison.png        # ReLU vs Parametric 性能對比
    ├── feature_importance.png            # 特徵重要性分析
    ├── decision_boundary.png             # PCA 決策邊界
    ├── calibration_curve.png             # 校準曲線
    ├── misclassification_analysis.png    # 誤分類分析
    ├── learning_rate_schedule.png        # 學習率調度曲線
    └── parameter_comparison.csv          # 參數變化對比表
```

## 🔑 關鍵實作特性

### ✅ 混合精度訓練（FP16）
- 使用 `torch.cuda.amp` 進行 FP16 訓練
- 針對 RTX A6000 優化，啟用 TensorFloat-32
- 顯著提升訓練速度和記憶體效率

### ✅ 參數化激活函數
- **ParametricTanh**: `output = α * tanh(β * x) + γ`
- **ParametricSigmoid**: `output = α * sigmoid(β * x) + γ`
- 參數可學習，自動優化激活函數形態
- Deep 模型前12層交替使用 Tanh/Sigmoid

### ✅ Early Stopping Warmup 機制
- **啟動延遲**：前 50 個 epoch 不檢查 early stopping
- **耐心值**：30 個 epoch
- **效果**：防止因初期震盪導致的過早停止

### ✅ 資料處理與訓練優化
- **切分**：80% 開發集 / 20% 測試集
- **交叉驗證**：5-Fold Stratified CV（保證類別平衡）
- **高效 DataLoader**：batch_size=512, 12 workers, pin_memory
- **學習率調度**：Cosine Annealing（實驗B）

### ✅ 完整視覺化分析
1. **5 指標訓練曲線**：Loss, Accuracy, AUC, Sensitivity, Specificity
2. **模型性能比較**：柱狀圖 + 箱線圖
3. **激活函數分析**：參數變化、形狀對比、分布圖
4. **決策邊界**：PCA 2D 投影
5. **校準曲線**：預測概率校準分析
6. **誤分類分析**：FP/FN 樣本詳細分析

## 🚀 如何使用

### 執行程式
```powershell
python mlp_advanced.py
```

### 查看結果
```powershell
# 查看訓練統計
python 查看結果.py

# 查看進階分析
python 查看進階結果.py
```

## 🛠️ 技術棧

- **深度學習框架**：PyTorch 2.x (CUDA 加速 + 混合精度)
- **資料處理**：pandas, NumPy, scikit-learn
- **視覺化**：Matplotlib, Seaborn
- **降維**：t-SNE, PCA (sklearn)
- **評估指標**：Accuracy, AUC, Sensitivity, Specificity

## 📈 實驗設計

### 實驗 A：模型架構比較（5-Fold CV）
- 比較 Shallow/Medium/Deep 三種架構
- 輸出平均訓練曲線（含標準差陰影）
- 輸出統計 CSV 和模型比較圖

### 實驗 B：進階分析（使用最佳模型）
- 標準 ReLU vs 參數化激活函數對比
- 完整的混淆矩陣、ROC 曲線分析
- 激活函數參數學習過程可視化
- 特徵重要性、決策邊界等進階分析

---

生成時間：2026-01-07  
執行環境：Python 3.x + CUDA + RTX A6000
