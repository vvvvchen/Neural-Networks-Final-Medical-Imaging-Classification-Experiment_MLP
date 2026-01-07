"""
快速查看進階版本的所有輸出結果
"""
import os
import webbrowser
from pathlib import Path

def list_all_outputs():
    """列出所有生成的檔案"""
    print("="*70)
    print("📊 進階版本輸出文件清單")
    print("="*70)
    
    results_dir = Path('results')
    
    # 1. Fold級別訓練圖
    folds_dir = results_dir / 'folds'
    if folds_dir.exists():
        fold_files = sorted(folds_dir.glob('*.png'))
        print(f"\n📁 Fold級別訓練圖 ({len(fold_files)}張)")
        print(f"   位置: {folds_dir}/")
        for f in fold_files:
            size_kb = f.stat().st_size / 1024
            print(f"   ✓ {f.name} ({size_kb:.1f} KB)")
    
    # 2. 統計CSV
    stats_dir = results_dir / 'statistics'
    if stats_dir.exists():
        csv_files = sorted(stats_dir.glob('*.csv'))
        print(f"\n📋 訓練統計CSV ({len(csv_files)}個)")
        print(f"   位置: {stats_dir}/")
        for f in csv_files:
            size_kb = f.stat().st_size / 1024
            print(f"   ✓ {f.name} ({size_kb:.1f} KB)")
            # 顯示CSV概要
            import pandas as pd
            df = pd.read_csv(f)
            print(f"      - 共 {len(df)} 個 epochs")
            print(f"      - 欄位: {len(df.columns)} 個（包含均值、標準差、最大最小值）")
    
    # 3. 平均訓練圖
    avg_files = sorted(results_dir.glob('*_history_avg.png'))
    if avg_files:
        print(f"\n📈 平均訓練圖（含±std陰影） ({len(avg_files)}張)")
        print(f"   位置: {results_dir}/")
        for f in avg_files:
            size_kb = f.stat().st_size / 1024
            print(f"   ✓ {f.name} ({size_kb:.1f} KB)")
    
    # 4. 進階分析
    advanced_dir = results_dir / 'advanced'
    if advanced_dir.exists():
        advanced_files = sorted(advanced_dir.glob('*.png'))
        print(f"\n🔬 進階分析圖表 ({len(advanced_files)}張)")
        print(f"   位置: {advanced_dir}/")
        analysis_types = {
            'learning_rate_schedule.png': '學習率調度曲線（Cosine Annealing）',
            'feature_importance.png': '特徵重要性分析（Top 20）',
            'decision_boundary.png': '決策邊界可視化（PCA 2D投影）',
            'calibration_curve.png': '校準曲線 + 預測概率分布',
            'misclassification_analysis.png': '誤分類樣本深度分析（4子圖）'
        }
        for f in advanced_files:
            size_kb = f.stat().st_size / 1024
            desc = analysis_types.get(f.name, '未知分析')
            print(f"   ✓ {f.name} ({size_kb:.1f} KB)")
            print(f"      → {desc}")
    
    # 統計總數
    total_images = len(list(results_dir.rglob('*.png')))
    total_csvs = len(list(results_dir.rglob('*.csv')))
    
    print(f"\n{'='*70}")
    print(f"📦 總計: {total_images} 張圖片 + {total_csvs} 個CSV文件")
    print(f"{'='*70}\n")


def show_csv_preview(csv_path, n_rows=10):
    """顯示CSV前幾行"""
    import pandas as pd
    
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*70}")
    print(f"CSV內容預覽: {os.path.basename(csv_path)}")
    print(f"{'='*70}")
    print(f"總行數: {len(df)} | 總列數: {len(df.columns)}")
    print(f"\n前 {n_rows} 行數據：")
    print(df.head(n_rows).to_string(index=False))
    
    # 顯示關鍵統計
    print(f"\n關鍵指標統計:")
    key_metrics = ['val_auc_mean', 'val_auc_std', 'val_acc_mean', 'val_loss_mean']
    for metric in key_metrics:
        if metric in df.columns:
            last_value = df[metric].iloc[-1]
            best_value = df[metric].max() if 'auc' in metric or 'acc' in metric else df[metric].min()
            print(f"  {metric:20s}: 最終={last_value:.4f}, 最佳={best_value:.4f}")


def open_sample_images():
    """打開範例圖片"""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    # 選擇要顯示的圖片
    samples = [
        'results/Shallow_history_avg.png',
        'results/advanced/feature_importance.png',
        'results/advanced/calibration_curve.png'
    ]
    
    existing = [s for s in samples if os.path.exists(s)]
    
    if not existing:
        print("⚠️ 圖片尚未生成")
        return
    
    print(f"\n開啟 {len(existing)} 張範例圖片...")
    for img_path in existing:
        try:
            os.startfile(img_path)
            print(f"  ✓ 已開啟: {img_path}")
        except:
            print(f"  ✗ 無法開啟: {img_path}")


def main():
    print("\n🚀 進階版本結果查看器\n")
    
    # 1. 列出所有文件
    list_all_outputs()
    
    # 2. 詢問是否查看CSV
    try:
        response = input("是否查看訓練統計CSV預覽？(y/n): ").strip().lower()
        if response == 'y':
            csv_files = list(Path('results/statistics').glob('*.csv'))
            for csv_file in csv_files:
                show_csv_preview(csv_file)
                print()
    except:
        pass
    
    # 3. 詢問是否打開圖片
    try:
        response = input("\n是否打開範例圖片？(y/n): ").strip().lower()
        if response == 'y':
            open_sample_images()
    except:
        pass
    
    print("\n" + "="*70)
    print("✨ 查看完畢！")
    print("="*70)
    print("\n提示：")
    print("  - 所有Fold訓練圖: results/folds/")
    print("  - 訓練統計CSV: results/statistics/")
    print("  - 平均訓練圖: results/*_history_avg.png")
    print("  - 進階分析: results/advanced/")
    print("\n詳細說明請參考: 進階版本說明.md")


if __name__ == '__main__':
    main()
