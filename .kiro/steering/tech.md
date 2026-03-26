# Technology Stack

## Language & Runtime
- Python 3.x (primary language for all components)
- No build system required (interpreted Python)

## Core Libraries

### Data Processing
- pandas >= 2.0.0 (data manipulation and feature engineering)
- numpy >= 1.24.0 (numerical operations)

### Machine Learning
- scikit-learn >= 1.3.0 (preprocessing, metrics, base models)
- xgboost >= 2.0.0 (gradient boosting classifier)
- lightgbm >= 4.0.0 (gradient boosting classifier)
- imbalanced-learn >= 0.11.0 (SMOTE and imbalance handling)
- optuna (hyperparameter optimization, imported in train_model.py)

### Explainability & Analysis
- shap >= 0.44.0 (model interpretation and feature importance)
- networkx >= 3.2.0 (graph analysis for fraud network detection)

### Visualization
- matplotlib >= 3.7.0 (static plots and graph visualizations)
- plotly >= 5.18.0 (interactive dashboard charts)
- streamlit >= 1.32.0 (web dashboard framework)

### API Integration
- requests >= 2.31.0 (data fetching from BitoPro API)

## Common Commands

### Full Pipeline Execution
```bash
# Run complete pipeline (fetch → feature engineering → graph → train → predict)
python run_pipeline.py

# Skip data fetching (use existing data/raw/)
python run_pipeline.py --skip-fetch

# Skip graph analysis (faster execution)
python run_pipeline.py --skip-graph

# Launch dashboard only
python run_pipeline.py --only-dashboard
```

### Individual Pipeline Steps
```bash
# Step 1: Fetch data from API
python src/data/fetch_data.py

# Step 2: Feature engineering
python src/data/feature_engineering.py

# Step 3: Graph analysis
python src/models/graph_analysis.py

# Step 4: Handle imbalance and split data
python src/data/handle_imbalance.py

# Step 5: Train models
python src/models/train_model.py

# Step 6: Generate SHAP explanations
python src/models/shap_explainer.py
```

### Dashboard
```bash
# Launch Streamlit dashboard
streamlit run app/dashboard/dashboard.py --server.port 8501
```

## API Integration

The system fetches data from BitoPro's PostgREST API:
- Base URL: `https://aws-event-api.bitopro.com`
- Authentication: None required (public endpoints)
- Pagination: PostgREST format (offset + limit)
- Headers: `Prefer: count=exact` for total count retrieval

## Model Artifacts

All trained models and outputs are saved to:
- `models/saved/` - Pickled model files (.pkl)
- `data/processed/` - Processed features and splits
- `outputs/reports/` - Analysis reports and predictions
- `outputs/plots/` - Visualization outputs
