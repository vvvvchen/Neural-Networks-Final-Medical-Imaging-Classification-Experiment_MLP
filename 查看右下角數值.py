"""
快速查看訓練圖表中的右下角數值摘要
"""
import os
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("📊 訓練圖表位置與說明")
    print("="*70)
    
    print("\n【單個Fold訓練圖】- 右下角顯示該fold的詳細訓練結果")
    print("位置: results/folds/\n")
    
    folds_dir = Path('results/folds')
    if folds_dir.exists():
        fold_files = sorted(folds_dir.glob('*.png'))
        for i, f in enumerate(fold_files, 1):
            print(f"  {i}. {f.name}")
            print(f"     → 右下角包含: 總回合數、5個最終指標、最佳AUC和準確率")
    
    print("\n【平均訓練圖】- 右下角顯示3個fold的平均統計")
    print("位置: results/\n")
    
    avg_files = sorted(Path('results').glob('*_history_avg.png'))
    for i, f in enumerate(avg_files, 1):
        print(f"  {i}. {f.name}")
        print(f"     → 右下角包含: 平均回合數、5個指標(均值±標準差)、最佳AUC")
    
    print("\n" + "="*70)
    print("💡 使用提示")
    print("="*70)
    print("""
1. 查看單個fold圖：了解每次訓練的具體情況
   - Total Epochs：該fold訓練了多少回合
   - Final Metrics：最後一個epoch的5個指標值
   - Best Metrics：訓練過程中的最佳表現

2. 查看平均圖：了解整體穩定性
   - 標準差小：3個fold表現一致，模型穩定
   - 標準差大：fold之間差異明顯，可能需要調參
   
3. 快速對比3個模型（Shallow, Medium, Deep）：
   - 直接看平均圖的右下角Summary
   - 比較Final AUC的均值
   - 查看標準差判斷哪個模型最穩定
""")
    
    # 詢問是否打開圖片
    print("="*70)
    try:
        choice = input("\n是否打開範例圖片查看右下角摘要？(y/n): ").strip().lower()
        if choice == 'y':
            # 打開幾個代表性的圖片
            samples = [
                'results/folds/Deep_fold1_history.png',  # 最佳模型的第一個fold
                'results/Deep_history_avg.png',          # 最佳模型的平均圖
            ]
            
            print("\n正在打開圖片...")
            for img_path in samples:
                if os.path.exists(img_path):
                    os.startfile(img_path)
                    print(f"  ✓ 已打開: {img_path}")
                else:
                    print(f"  ✗ 文件不存在: {img_path}")
            
            print("\n請查看圖片右下角的Summary文字框！")
    except:
        pass
    
    print("\n" + "="*70)
    print("📖 詳細說明請參考: 右下角數值說明.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
