"""
快速開啟結果報告
"""
import webbrowser
import os

html_path = os.path.abspath('results/report.html')

print("="*70)
print("🧠 MLP 二元分類專案 - 結果查看")
print("="*70)
print("\n📊 實驗結果摘要")
print("-"*70)
print("✅ 最佳模型: Deep MLP")
print("✅ 測試集 AUC: 0.9164")
print("✅ 測試集準確率: 0.8435 (84.35%)")
print("✅ 靈敏度: 0.8790 (87.90%)")
print("✅ 特異度: 0.8080 (80.80%)")
print()
print("📁 生成的檔案")
print("-"*70)
print("1. results/report.html - 繁體中文完整報告（即將開啟）")
print("2. results/Shallow_history.png - 淺層模型訓練圖（5 指標）")
print("3. results/Medium_history.png - 中層模型訓練圖（5 指標）")
print("4. results/Deep_history.png - 深層模型訓練圖（5 指標）")
print("5. results/final_evaluation.png - 測試集評估圖")
print("6. 專案說明.md - 詳細技術文件（繁體中文）")
print()
print("📖 說明文件")
print("-"*70)
print("• 專案說明.md - 完整的繁體中文技術說明與使用指南")
print("• README.md - 英文專案文件")
print()
print("🌐 正在開啟 HTML 報告...")
print("="*70)

webbrowser.open(f'file:///{html_path}')
