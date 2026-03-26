# Project Structure

## Directory Organization

```
bitoguard/
├── app/
│   └── dashboard/
│       └── dashboard.py          # Streamlit web dashboard
├── data/
│   ├── raw/                      # Raw data from API (7 CSV files)
│   └── processed/                # Engineered features and train/test splits
├── models/
│   └── saved/                    # Trained model artifacts (.pkl files)
├── outputs/
│   ├── reports/                  # Analysis reports and predictions
│   └── plots/                    # Graph visualizations
├── src/
│   ├── data/
│   │   ├── fetch_data.py         # API data fetching (PostgREST)
│   │   ├── feature_engineering.py # Feature creation (25-28 features)
│   │   └── handle_imbalance.py   # Data splitting and scaling
│   └── models/
│       ├── graph_analysis.py     # NetworkX graph features
│       ├── train_model.py        # Model training with Optuna
│       └── shap_explainer.py     # SHAP-based risk diagnosis
├── run_pipeline.py               # Orchestrator script
├── requirements.txt              # Python dependencies
└── README.md
```

## Pipeline Flow

The system follows a strict sequential pipeline:

1. **fetch_data.py** → Pulls 7 tables from BitoPro API into `data/raw/`
2. **feature_engineering.py** → Creates 25-28 interpretable features, outputs to `data/processed/features.csv`
3. **graph_analysis.py** → Builds NetworkX graph, computes graph features, merges back to features.csv
4. **handle_imbalance.py** → Splits train/test, applies RobustScaler, outputs train_balanced.csv and test.csv
5. **train_model.py** → Trains ensemble models with Optuna tuning, saves to `models/saved/`
6. **shap_explainer.py** → Generates risk diagnosis reports with SHAP values
7. **dashboard.py** → Visualizes results in Streamlit web interface

## Key Data Files

### Raw Data (data/raw/)
- `user_info.csv` - User profiles and KYC data
- `twd_transfer.csv` - TWD (fiat) deposits and withdrawals
- `crypto_transfer.csv` - Crypto deposits and withdrawals
- `usdt_twd_trading.csv` - USDT/TWD trading records
- `usdt_swap.csv` - USDT swap transactions
- `train_label.csv` - Training labels (status: 0=normal, 1=blacklist)
- `predict_label.csv` - Prediction target user IDs

### Processed Data (data/processed/)
- `features.csv` - Full feature matrix with labels
- `predict_features.csv` - Features for prediction set
- `train_balanced.csv` - Scaled training set
- `test.csv` - Scaled test set
- `feature_cols.json` - List of feature column names
- `class_weights.csv` - Class imbalance weights
- `scaler_info.csv` - RobustScaler parameters

### Model Outputs (models/saved/)
- `f1_optimized_model.pkl` - Final ensemble model with optimal threshold
- Individual model files: `lightgbm.pkl`, `xgboost.pkl`, etc.

## Code Conventions

### Language
- Primary language: Traditional Chinese (繁體中文) for comments and print statements
- Variable names: English
- Documentation strings: Chinese with technical terms in English

### Naming Patterns
- Feature columns: lowercase with underscores (e.g., `twd_deposit_total`, `kyc_rushed`)
- Graph features: prefixed with `graph_` (e.g., `graph_degree`, `graph_pagerank`)
- File paths: Use `pathlib.Path` for cross-platform compatibility
- Constants: UPPERCASE with underscores (e.g., `PROCESSED_DIR`, `RANDOM_STATE`)

### Data Handling
- Always use `low_memory=False` when reading CSVs with mixed types
- Use `errors="coerce"` for datetime parsing to handle invalid dates
- Fill missing values with 0 for numerical features
- Use `encoding="utf-8-sig"` when saving CSVs to preserve Chinese characters

### Path Resolution
- All paths relative to project root (bitoguard/)
- Dashboard uses `Path(__file__).resolve().parents[2]` to locate project root
- Create directories with `mkdir(parents=True, exist_ok=True)`

### Error Handling
- Use `warnings.filterwarnings("ignore")` to suppress sklearn/pandas warnings
- Check file existence with `Path.exists()` before reading
- Provide informative error messages in Chinese

### Chinese Font Configuration
- For matplotlib plots, always set Chinese font at the beginning:
  ```python
  plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
  plt.rcParams['axes.unicode_minus'] = False
  ```
- For Streamlit/Plotly charts, use CSS font-family:
  ```python
  fig.update_layout(font=dict(family="Microsoft JhengHei, Noto Sans TC, sans-serif"))
  ```
- This prevents Chinese characters from appearing as boxes (□□□)

## Anti-Patterns to Avoid

### Data Leakage Prevention
- NEVER use test set labels during feature engineering
- NEVER fit scalers on combined train+test data
- Graph features must use `train_blacklist` only (not full blacklist)
- Always split data BEFORE any preprocessing that learns from data

### Performance
- Filter high-frequency groups (> 50 users per IP/wallet) to avoid NAT/proxy noise
- Pre-compute connected components once, not per-node
- Use adjacency dictionaries instead of repeated `G.neighbors()` calls
- Limit visualization subgraphs to prevent memory issues

### Model Training
- Use `scale_pos_weight` parameter for class imbalance (not just SMOTE)
- Optimize for F1 score directly in Optuna objective (not just logloss)
- Use StratifiedKFold to maintain class distribution in CV
- Find optimal threshold on validation set using fine-grained search (0.01 steps)
