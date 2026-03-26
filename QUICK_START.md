# BitoGuard 快速啟動指南

## 🚀 啟動 Dashboard

```bash
# 方法 1：使用 run_pipeline.py（推薦）
python run_pipeline.py --only-dashboard

# 方法 2：直接啟動 Streamlit
streamlit run app/dashboard/dashboard.py --server.port 8502
```

Dashboard 網址：**http://localhost:8502**

## 🎨 重新生成圖表

如果圖表中的中文顯示為方框，執行：

```bash
python regenerate_plots.py
```

這會重新生成：
- 關聯圖譜（graph_full_network.png, graph_blacklist_neighborhood.png）
- SHAP 特徵重要性圖（shap_summary.png, shap_importance.png）
- 個別用戶 Waterfall 圖（waterfall_user_*.png）

## 📊 完整 Pipeline 執行

```bash
# 完整執行（包含 API 拉取）
python run_pipeline.py

# 跳過 API 拉取（使用現有資料）
python run_pipeline.py --skip-fetch

# 跳過圖譜分析（加快速度）
python run_pipeline.py --skip-graph
```

## 🔧 個別步驟執行

```bash
# 步驟 1：拉取資料
python src/data/fetch_data.py

# 步驟 2：特徵工程
python src/data/feature_engineering.py

# 步驟 3：圖譜分析
python src/models/graph_analysis.py

# 步驟 4：資料切分與縮放
python src/data/handle_imbalance.py

# 步驟 5：模型訓練
python src/models/train_model.py

# 步驟 6：SHAP 解釋
python src/models/shap_explainer.py
```

## 📁 重要檔案位置

### 輸入資料
- `data/raw/` - API 拉取的原始資料（7 個 CSV 檔）
- `data/processed/` - 處理後的特徵與訓練/測試集

### 輸出結果
- `outputs/reports/` - 分析報告與預測結果
  - `user_risk_scores.csv` - 用戶風險分數
  - `risk_diagnosis.json` - 風險診斷書
  - `submission.csv` - 最終提交檔案
- `outputs/plots/` - 所有視覺化圖表
- `models/saved/` - 訓練好的模型檔案

## ⚠️ 常見問題

### Dashboard 無法啟動
- 確認 port 8502 沒有被佔用
- 檢查 `outputs/reports/` 是否有必要檔案
- 確認已安裝所有套件：`pip install -r requirements.txt`

### 中文顯示為方框
- 執行 `python regenerate_plots.py` 重新生成圖表
- 確認系統已安裝微軟正黑體（Microsoft JhengHei）

### Pipeline 執行失敗
- 確認 `data/raw/` 有完整的 7 個 CSV 檔案
- 檢查每個步驟的錯誤訊息
- 可以單獨執行失敗的步驟進行除錯
