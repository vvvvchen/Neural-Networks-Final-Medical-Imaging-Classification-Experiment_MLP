# 激活函數參數配置說明

## 調整日期
2026年1月7日

## 參數說明
- **α (alpha)**: 縮放參數，控制輸出範圍的大小
- **β (beta)**: 陡峭度參數，控制函數的斜率（輸入縮放）
- **γ (gamma)**: 偏移參數，控制函數的垂直位移

## 當前參數配置（對應 mlp_advanced.py）

### Deep 模型 - 前12層參數化激活

| Layer | 類型 | α (縮放) | β (陡峭度) | γ (偏移) |
|-------|------|---------|-----------|---------|
| 1 | ParametricTanh | 1.5 | 1.2 | 0.1 |
| 2 | ParametricSigmoid | 1.8 | 1.5 | -0.3 |
| 3 | ParametricTanh | 1.3 | 1.1 | 0.05 |
| 4 | ParametricSigmoid | 1.6 | 1.3 | -0.25 |
| 5 | ParametricTanh | 1.2 | 1.0 | 0.08 |
| 6 | ParametricSigmoid | 1.7 | 1.4 | -0.35 |
| 7 | ParametricTanh | 1.4 | 1.15 | 0.06 |
| 8 | ParametricSigmoid | 1.5 | 1.2 | -0.28 |
| 9 | ParametricTanh | 1.25 | 1.05 | 0.04 |
| 10 | ParametricSigmoid | 1.4 | 1.1 | -0.22 |
| 11 | ParametricTanh | 1.1 | 0.95 | 0.02 |
| 12 | ParametricSigmoid | 1.3 | 1.0 | -0.18 |

**Layer 13-19**: 標準 ReLU  
**Layer 20**: 輸出層（BCEWithLogitsLoss）

---

### Medium 模型 - 前6層參數化激活

| Layer | 類型 | α (縮放) | β (陡峭度) | γ (偏移) |
|-------|------|---------|-----------|---------|
| 1 | ParametricTanh | 1.5 | 1.2 | 0.1 |
| 2 | ParametricSigmoid | 1.8 | 1.5 | -0.3 |
| 3 | ParametricTanh | 1.3 | 1.1 | 0.05 |
| 4 | ParametricSigmoid | 1.6 | 1.3 | -0.25 |
| 5 | ParametricTanh | 1.2 | 1.0 | 0.08 |
| 6 | ParametricSigmoid | 1.7 | 1.4 | -0.35 |

**Layer 7-9**: 標準 ReLU  
**Layer 10**: 輸出層

---

### Shallow 模型 - 前3層參數化激活

| Layer | 類型 | α (縮放) | β (陡峭度) | γ (偏移) |
|-------|------|---------|-----------|---------|
| 1 | ParametricTanh | 1.5 | 1.2 | 0.1 |
| 2 | ParametricSigmoid | 1.8 | 1.5 | -0.3 |
| 3 | ParametricTanh | 1.3 | 1.1 | 0.05 |

**Layer 4**: 標準 ReLU  
**Layer 5**: 輸出層

---

## 調整策略總結

### 1. **增強非線性表達能力**
   - α 值範圍 1.1 ~ 1.8（擴大輸出範圍）
   - 使模型能學習到更豐富的特徵表示

### 2. **優化梯度流動**
   - β 值範圍 0.95 ~ 1.5（控制陡峭度）
   - 改善深層網絡的梯度傳播

### 3. **平衡激活分布**
   - Tanh 層：小正偏移（γ = 0.02 ~ 0.1）
   - Sigmoid 層：負偏移（γ = -0.35 ~ -0.18）

### 4. **漸進式調整**
   - 前幾層調整幅度較大（學習底層特徵）
   - 後面層調整幅度較小（保持穩定性）

---

## 實驗結果查看

執行訓練後，可以在以下位置查看結果：

| 檔案 | 說明 |
|------|------|
| `results/advanced/parameter_changes.png` | 參數變化柱狀圖 |
| `results/advanced/activation_shapes.png` | 激活函數形狀對比 |
| `results/advanced/activation_distribution_comparison.png` | 激活值分布對比 |
| `results/advanced/performance_comparison.png` | ReLU vs Parametric 性能對比 |
| `results/advanced/parameter_comparison.csv` | 參數對比詳細表格 |

---

## 程式碼參考

```python
# mlp_advanced.py 中 DeepMLP 的參數化激活配置
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
```

---

**建立時間**: 2026-01-07  
**對應程式**: `mlp_advanced.py`  
**框架**: PyTorch 2.x
