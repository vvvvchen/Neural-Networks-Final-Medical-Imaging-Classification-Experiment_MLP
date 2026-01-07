"""
激活函數分析工具
提供參數變化可視化和對比分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


def plot_parameter_changes(initial_params, final_params, save_path):
    """繪製激活函數參數變化圖 - Before/After 對比"""
    print("\n繪製參數變化對比圖...")
    
    # 提取所有層的參數
    layers = sorted([name for name in initial_params.keys()])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Activation Function Parameters: Before vs After Training', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    param_names = ['alpha', 'beta', 'gamma']
    param_labels = ['α (Scale)', 'β (Steepness)', 'γ (Shift)']
    colors_before = ['#87CEEB', '#FFB6C1', '#90EE90']  # 淺藍、淺紅、淺綠
    colors_after = ['#1E90FF', '#DC143C', '#228B22']   # 深藍、深紅、深綠
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[idx]
        
        # 收集 before/after 數據
        before_vals = [initial_params[layer][param] for layer in layers]
        after_vals = [final_params[layer][param] for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        # 繪製柱狀圖
        bars1 = ax.bar(x - width/2, before_vals, width, label='Before Training',
                      color=colors_before[idx], alpha=0.9, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, after_vals, width, label='After Training',
                      color=colors_after[idx], alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # 添加數值標籤
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold')
        
        # 設置
        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'Parameter {label} Changes', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{i+1}' for i in range(len(layers))], fontsize=9)
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存參數變化圖: {save_path}")


def plot_activation_function_shapes(initial_params, final_params, save_path):
    """繪製激活函數形狀變化"""
    print("\n繪製激活函數形狀變化...")
    
    layers = sorted([name for name in initial_params.keys()])[:6]  # 只顯示前6層
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Activation Function Shapes: Before (dashed) vs After (solid)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    x = np.linspace(-5, 5, 1000)
    
    for idx, layer in enumerate(layers):
        ax = axes[idx // 3, idx % 3]
        
        # Before
        alpha_b = initial_params[layer]['alpha']
        beta_b = initial_params[layer]['beta']
        gamma_b = initial_params[layer]['gamma']
        act_type = initial_params[layer]['type']
        
        # After
        alpha_a = final_params[layer]['alpha']
        beta_a = final_params[layer]['beta']
        gamma_a = final_params[layer]['gamma']
        
        # 計算輸出
        if act_type == 'Tanh':
            y_before = alpha_b * np.tanh(beta_b * x) + gamma_b
            y_after = alpha_a * np.tanh(beta_a * x) + gamma_a
            func_name = 'Tanh'
            color = '#3498db'
        else:  # Sigmoid
            y_before = alpha_b / (1 + np.exp(-beta_b * x)) + gamma_b
            y_after = alpha_a / (1 + np.exp(-beta_a * x)) + gamma_a
            func_name = 'Sigmoid'
            color = '#e74c3c'
        
        # 繪製
        ax.plot(x, y_before, '--', linewidth=2.5, color=color, alpha=0.5, 
               label=f'Before: α={alpha_b:.2f}, β={beta_b:.2f}, γ={gamma_b:.2f}')
        ax.plot(x, y_after, '-', linewidth=2.5, color=color, 
               label=f'After: α={alpha_a:.2f}, β={beta_a:.2f}, γ={gamma_a:.2f}')
        
        # 參考線
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Input', fontsize=10)
        ax.set_ylabel('Output', fontsize=10)
        ax.set_title(f'Layer {idx+1} ({func_name})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([-5, 5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存激活函數形狀圖: {save_path}")


def save_parameter_comparison_table(initial_params, final_params, save_path):
    """保存參數對比表為CSV"""
    print("\n生成參數對比表...")
    
    layers = sorted([name for name in initial_params.keys()])
    
    data = []
    for layer in layers:
        layer_num = layer.split('act')[-1]
        act_type = initial_params[layer]['type']
        
        # Before
        alpha_b = initial_params[layer]['alpha']
        beta_b = initial_params[layer]['beta']
        gamma_b = initial_params[layer]['gamma']
        
        # After
        alpha_a = final_params[layer]['alpha']
        beta_a = final_params[layer]['beta']
        gamma_a = final_params[layer]['gamma']
        
        # 變化量
        alpha_change = alpha_a - alpha_b
        beta_change = beta_a - beta_b
        gamma_change = gamma_a - gamma_b
        
        # 變化百分比
        alpha_pct = (alpha_change / alpha_b * 100) if alpha_b != 0 else 0
        beta_pct = (beta_change / beta_b * 100) if beta_b != 0 else 0
        gamma_pct = (gamma_change / abs(gamma_b) * 100) if gamma_b != 0 else 0
        
        data.append({
            'Layer': f'Layer {layer_num}',
            'Activation': act_type,
            'α_Before': alpha_b,
            'α_After': alpha_a,
            'α_Change': alpha_change,
            'α_Change(%)': alpha_pct,
            'β_Before': beta_b,
            'β_After': beta_a,
            'β_Change': beta_change,
            'β_Change(%)': beta_pct,
            'γ_Before': gamma_b,
            'γ_After': gamma_a,
            'γ_Change': gamma_change,
            'γ_Change(%)': gamma_pct
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, float_format='%.4f', encoding='utf-8-sig')
    print(f"✓ 已儲存參數對比表: {save_path}")
    
    # 同時輸出到控制台
    print("\n" + "="*70)
    print("參數變化總結表")
    print("="*70)
    for _, row in df.iterrows():
        print(f"\n{row['Layer']} ({row['Activation']})")
        print(f"  α: {row['α_Before']:.3f} → {row['α_After']:.3f} (Δ{row['α_Change']:+.3f}, {row['α_Change(%)']:+.1f}%)")
        print(f"  β: {row['β_Before']:.3f} → {row['β_After']:.3f} (Δ{row['β_Change']:+.3f}, {row['β_Change(%)']:+.1f}%)")
        print(f"  γ: {row['γ_Before']:.3f} → {row['γ_After']:.3f} (Δ{row['γ_Change']:+.3f}, {row['γ_Change(%)']:+.1f}%)")
    print("="*70)


def plot_performance_comparison(results_comparison, save_path):
    """繪製ReLU vs Parametric性能對比"""
    print("\n繪製性能對比圖...")
    
    metrics_relu = results_comparison['ReLU']['metrics']
    metrics_param = results_comparison['Parametric']['metrics']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance Comparison: ReLU vs Parametric Activation', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # 左圖：柱狀圖對比
    ax1 = axes[0]
    metrics_names = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    metrics_keys = ['auc', 'acc', 'sens', 'spec']
    
    relu_vals = [metrics_relu[key] for key in metrics_keys]
    param_vals = [metrics_param[key] for key in metrics_keys]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, relu_vals, width, label='ReLU',
                   color='#87CEEB', alpha=0.9, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x + width/2, param_vals, width, label='Parametric',
                   color='#90EE90', alpha=0.9, edgecolor='black', linewidth=2)
    
    # 添加數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax1.set_title('Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=11)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_ylim([0.8, 1.0])
    ax1.grid(alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右圖：改善百分比
    ax2 = axes[1]
    improvements = [(param_vals[i] - relu_vals[i]) / relu_vals[i] * 100 
                   for i in range(len(metrics_keys))]
    
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax2.barh(metrics_names, improvements, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    # 添加數值標籤
    for bar, val in zip(bars, improvements):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:+.2f}%', ha='left' if val > 0 else 'right', 
                va='center', fontsize=11, fontweight='bold')
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_xlabel('Improvement (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Parametric vs ReLU Improvement', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存性能對比圖: {save_path}")


def plot_activation_comparison(model_relu, model_param, test_loader, device, save_path):
    """對比ReLU和Parametric模型的激活值分布"""
    print("\n分析激活值分布...")
    
    model_relu.eval()
    model_param.eval()
    
    # 收集激活值
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            # ReLU模型
            _ = model_relu(X, record_activations=True)
            # Parametric模型
            _ = model_param(X, record_activations=True)
            break  # 只取一個batch
    
    # 選擇前6層進行可視化
    layers_to_plot = [f'layer{i}' for i in range(1, 7)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Activation Distribution: ReLU vs Parametric', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, layer in enumerate(layers_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        if layer in model_relu.activation_outputs and layer in model_param.activation_outputs:
            relu_acts = model_relu.activation_outputs[layer].numpy().flatten()
            param_acts = model_param.activation_outputs[layer].numpy().flatten()
            
            # 繪製分布
            ax.hist(relu_acts, bins=50, alpha=0.6, label='ReLU', 
                   color='#87CEEB', edgecolor='black', density=True)
            ax.hist(param_acts, bins=50, alpha=0.6, label='Parametric', 
                   color='#90EE90', edgecolor='black', density=True)
            
            # 統計資訊
            relu_mean, relu_std = np.mean(relu_acts), np.std(relu_acts)
            param_mean, param_std = np.mean(param_acts), np.std(param_acts)
            
            stats_text = f'ReLU: μ={relu_mean:.2f}, σ={relu_std:.2f}\n'
            stats_text += f'Param: μ={param_mean:.2f}, σ={param_std:.2f}'
            
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('Activation Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Layer {idx+1}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 已儲存激活值分布圖: {save_path}")
