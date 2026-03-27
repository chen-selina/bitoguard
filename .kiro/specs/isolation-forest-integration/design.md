# Design Document: Isolation Forest Integration

## Overview

This design replaces CatBoost with Isolation Forest in BitoGuard's ensemble fraud detection system to enable detection of novel fraud patterns through unsupervised learning. The current system uses supervised models (XGBoost, LightGBM, CatBoost) that can only recognize fraud patterns similar to historical training examples. By integrating Isolation Forest, an unsupervised anomaly detection algorithm, the system gains the ability to flag users whose behavior significantly deviates from normal patterns, even if those specific fraud schemes haven't been seen before.

### Key Design Decisions

1. **Three-Model Ensemble**: XGBoost + LightGBM + Isolation Forest (removing CatBoost)
2. **Unified Feature Space**: All models operate on the same 25-28 engineered features
3. **Stacking Architecture**: LogisticRegression meta-model combines base model predictions
4. **Anomaly Score Calibration**: Transform Isolation Forest's decision_function output to [0,1] probability-like scores
5. **Consistent Cross-Validation**: Use StratifiedKFold (N_SPLITS=5) for all models to generate OOF predictions

### Research Findings

**Isolation Forest Algorithm**:
- Unsupervised tree-based anomaly detection that isolates outliers through random partitioning
- Anomalies require fewer splits to isolate (shorter path length in trees)
- Returns decision_function scores (negative for anomalies, positive for normal)
- Key hyperparameters: n_estimators (number of trees), max_samples (subsample size), contamination (expected anomaly ratio), max_features (feature subsample)

**Integration with Supervised Models**:
- Isolation Forest provides complementary signal: structural anomalies vs. learned fraud patterns
- Stacking allows meta-model to learn optimal weighting between supervised and unsupervised signals
- Calibration is critical: decision_function outputs must be transformed to probability-like [0,1] scale for fair comparison with XGBoost/LightGBM probabilities

**Hyperparameter Optimization**:
- Optuna can optimize Isolation Forest by evaluating how well anomaly scores correlate with fraud labels in cross-validation
- Objective function: maximize F1 score using optimal threshold on validation set
- This bridges unsupervised algorithm with supervised evaluation metric

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Data Loading & Preprocessing                                │
│  - Load train_balanced.csv, test.csv                         │
│  - Extract X_train, y_train, X_test, y_test                 │
│  - Feature space: 25-28 dimensions                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Hyperparameter Optimization (Optuna, 50 trials each)       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  LightGBM   │  │   XGBoost   │  │  Isolation  │        │
│  │   Tuning    │  │   Tuning    │  │   Forest    │        │
│  │             │  │             │  │   Tuning    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                  │
│         └────────────────┴────────────────┘                  │
│                          │                                    │
│              Objective: Max F1 Score                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5-Fold Cross-Validation (StratifiedKFold)                   │
│                                                               │
│  For each fold:                                               │
│    ┌─────────────────────────────────────────────┐          │
│    │  Train base models on train_idx              │          │
│    │  Generate predictions on val_idx             │          │
│    │  Accumulate OOF predictions                  │          │
│    │  Generate test predictions                   │          │
│    └─────────────────────────────────────────────┘          │
│                                                               │
│  Output:                                                      │
│    - lgbm_oof, xgb_oof, iforest_oof (train set)             │
│    - lgbm_test_probs, xgb_test_probs, iforest_test_probs    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Anomaly Score Calibration (Isolation Forest only)           │
│                                                               │
│  For each fold's Isolation Forest predictions:               │
│    1. Get decision_function scores                           │
│    2. Find min_score, max_score                              │
│    3. Normalize: (score - min) / (max - min)                 │
│    4. Invert: 1 - normalized (higher = more anomalous)       │
│                                                               │
│  Result: iforest_oof and iforest_test_probs in [0,1]        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Meta-Model Training (Stacking)                              │
│                                                               │
│  X_stack_train = [lgbm_oof, xgb_oof, iforest_oof]           │
│  X_stack_test  = [mean(lgbm_test), mean(xgb_test),          │
│                   mean(iforest_test)]                        │
│                                                               │
│  meta_model = LogisticRegression()                           │
│  meta_model.fit(X_stack_train, y_train)                     │
│                                                               │
│  oof_final_prob = meta_model.predict_proba(X_stack_train)   │
│  final_prob = meta_model.predict_proba(X_stack_test)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Threshold Optimization                                       │
│                                                               │
│  best_thresh = find_best_threshold_refined(                  │
│      y_train, oof_final_prob, thresholds=0.1 to 0.9, step=0.01) │
│                                                               │
│  Maximize F1 score on OOF predictions                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Final Evaluation & Model Persistence                        │
│                                                               │
│  y_pred = (final_prob >= best_thresh).astype(int)           │
│                                                               │
│  Metrics: Precision, Recall, F1, Confusion Matrix           │
│                                                               │
│  Save:                                                        │
│    - models/saved/f1_optimized_model.pkl                     │
│    - models/saved/isolationforest.pkl                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: `data/processed/train_balanced.csv`, `data/processed/test.csv`
2. **Feature Extraction**: 25-28 numerical features (already scaled by RobustScaler)
3. **Base Model Training**: Each model trained independently on same features
4. **OOF Generation**: StratifiedKFold ensures each training sample gets exactly one OOF prediction
5. **Score Calibration**: Isolation Forest scores transformed to [0,1] range
6. **Stacking**: Meta-model learns to combine three base model outputs
7. **Threshold Optimization**: Fine-grained search (0.01 steps) on OOF predictions
8. **Output**: Final ensemble model + optimal threshold saved to disk

## Components and Interfaces

### 1. Data Loading Module

**Function**: `load_and_preprocess()`

**Inputs**:
- `TRAIN_PATH`: Path to train_balanced.csv
- `TEST_PATH`: Path to test.csv
- `FEAT_COLS_PATH`: Path to feature_cols.json

**Outputs**:
- `X_train`: numpy array (n_samples, n_features)
- `y_train`: numpy array (n_samples,)
- `X_test`: numpy array (n_test, n_features)
- `y_test`: numpy array (n_test,)
- `feature_cols`: list of feature names
- `train_spw`: float, scale_pos_weight ratio (neg_count / pos_count)

**Processing**:
- Load CSV files with `low_memory=False`
- Handle categorical features with LabelEncoder (if any remain after feature engineering)
- Extract feature columns from JSON or infer from dataframe
- Calculate class imbalance ratio for supervised models

### 2. Hyperparameter Optimization Module

**Function**: `tune_hyperparams_v2(X_train, y_train, train_spw, model_type)`

**Inputs**:
- `X_train`: Training features
- `y_train`: Training labels
- `train_spw`: Scale positive weight for supervised models
- `model_type`: "lgbm", "xgb", or "iforest"

**Outputs**:
- `best_params`: dict of optimized hyperparameters

**Hyperparameter Ranges**:

**LightGBM**:
- n_estimators: [500, 1000]
- learning_rate: [0.005, 0.05] (log scale)
- num_leaves: [31, 128]
- feature_fraction: [0.7, 1.0]
- bagging_fraction: [0.7, 1.0]
- bagging_freq: [1, 7]
- scale_pos_weight: [1.0, train_spw]
- lambda_l1: [1e-8, 10.0] (log scale)
- lambda_l2: [1e-8, 10.0] (log scale)

**XGBoost**:
- n_estimators: [500, 1000]
- max_depth: [4, 10]
- learning_rate: [0.005, 0.05] (log scale)
- subsample: [0.7, 1.0]
- colsample_bytree: [0.7, 1.0]
- scale_pos_weight: [1.0, train_spw]

**Isolation Forest** (NEW):
- n_estimators: [100, 500]
- max_samples: [0.5, 1.0] or 'auto'
- contamination: [0.01, 0.3]
- max_features: [0.5, 1.0]
- random_state: 42 (fixed)

**Optimization Process**:
- Use 3-fold StratifiedKFold for speed (inner CV)
- For each trial:
  - Train model with suggested hyperparameters
  - Generate predictions on validation fold
  - Find optimal threshold for that fold
  - Calculate F1 score
  - Average F1 across 3 folds
- Optuna maximizes mean F1 score
- Run 50 trials per model

### 3. Isolation Forest Training Module

**Function**: `train_isolation_forest(X_train, y_train, params)`

**Inputs**:
- `X_train`: Training features (labels not used for training, only evaluation)
- `y_train`: Training labels (for evaluation only)
- `params`: Hyperparameters from Optuna

**Outputs**:
- `model`: Trained IsolationForest instance
- `oof_scores`: Calibrated anomaly scores [0,1] for training set
- `test_scores`: Calibrated anomaly scores [0,1] for test set

**Processing**:
1. Initialize IsolationForest with params
2. Fit on X_train (unsupervised, ignores y_train)
3. Generate decision_function scores
4. Calibrate scores to [0,1] range
5. Return model and calibrated scores

### 4. Anomaly Score Calibration Module

**Function**: `calibrate_iforest_scores(decision_scores)`

**Inputs**:
- `decision_scores`: numpy array of decision_function outputs (can be negative)

**Outputs**:
- `calibrated_scores`: numpy array in [0,1] range, higher = more anomalous

**Algorithm**:
```python
min_score = decision_scores.min()
max_score = decision_scores.max()
normalized = (decision_scores - min_score) / (max_score - min_score + 1e-9)
# Invert because decision_function returns negative for anomalies
calibrated = 1 - normalized
return calibrated
```

**Rationale**:
- Isolation Forest's decision_function returns negative values for anomalies
- We need [0,1] range where 1 = high fraud risk (consistent with XGBoost/LightGBM probabilities)
- Min-max normalization preserves relative ordering
- Inversion ensures higher scores = higher risk

### 5. Cross-Validation Module

**Function**: `generate_oof_predictions(X_train, y_train, X_test, params_dict)`

**Inputs**:
- `X_train`, `y_train`: Training data
- `X_test`: Test data
- `params_dict`: dict with keys 'lgbm', 'xgb', 'iforest' containing hyperparameters

**Outputs**:
- `lgbm_oof`, `xgb_oof`, `iforest_oof`: OOF predictions on training set
- `lgbm_test_probs`, `xgb_test_probs`, `iforest_test_probs`: List of 5 test predictions per model

**Processing**:
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgbm_oof = np.zeros(len(y_train))
xgb_oof = np.zeros(len(y_train))
iforest_oof = np.zeros(len(y_train))

lgbm_test_probs = []
xgb_test_probs = []
iforest_test_probs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    # LightGBM
    m_lgb = lgb.LGBMClassifier(**params_dict['lgbm'])
    m_lgb.fit(X_train[train_idx], y_train[train_idx])
    lgbm_oof[val_idx] = m_lgb.predict_proba(X_train[val_idx])[:, 1]
    lgbm_test_probs.append(m_lgb.predict_proba(X_test)[:, 1])
    
    # XGBoost
    m_xgb = xgb.XGBClassifier(**params_dict['xgb'])
    m_xgb.fit(X_train[train_idx], y_train[train_idx])
    xgb_oof[val_idx] = m_xgb.predict_proba(X_train[val_idx])[:, 1]
    xgb_test_probs.append(m_xgb.predict_proba(X_test)[:, 1])
    
    # Isolation Forest
    m_if = IsolationForest(**params_dict['iforest'])
    m_if.fit(X_train[train_idx])  # Unsupervised: no labels
    
    # Calibrate validation scores
    val_decision = m_if.decision_function(X_train[val_idx])
    iforest_oof[val_idx] = calibrate_iforest_scores(val_decision)
    
    # Calibrate test scores
    test_decision = m_if.decision_function(X_test)
    iforest_test_probs.append(calibrate_iforest_scores(test_decision))

return (lgbm_oof, xgb_oof, iforest_oof, 
        lgbm_test_probs, xgb_test_probs, iforest_test_probs)
```

### 6. Meta-Model (Stacking) Module

**Function**: `train_meta_model(oof_predictions, y_train, test_predictions)`

**Inputs**:
- `oof_predictions`: tuple of (lgbm_oof, xgb_oof, iforest_oof)
- `y_train`: Training labels
- `test_predictions`: tuple of (lgbm_test_list, xgb_test_list, iforest_test_list)

**Outputs**:
- `meta_model`: Trained LogisticRegression instance
- `oof_final_prob`: Meta-model predictions on training set
- `final_prob`: Meta-model predictions on test set

**Processing**:
```python
# Stack OOF predictions
X_stack_train = np.column_stack(oof_predictions)

# Average test predictions across 5 folds
lgbm_test_mean = np.mean(test_predictions[0], axis=0)
xgb_test_mean = np.mean(test_predictions[1], axis=0)
iforest_test_mean = np.mean(test_predictions[2], axis=0)
X_stack_test = np.column_stack([lgbm_test_mean, xgb_test_mean, iforest_test_mean])

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(X_stack_train, y_train)

# Generate final predictions
oof_final_prob = meta_model.predict_proba(X_stack_train)[:, 1]
final_prob = meta_model.predict_proba(X_stack_test)[:, 1]

return meta_model, oof_final_prob, final_prob
```

### 7. Threshold Optimization Module

**Function**: `find_best_threshold_refined(y_true, y_prob, name="Model")`

**Inputs**:
- `y_true`: True labels
- `y_prob`: Predicted probabilities
- `name`: Model name for logging

**Outputs**:
- `best_thresh`: Optimal threshold value
- `best_f1`: F1 score at optimal threshold

**Algorithm**:
```python
thresholds = np.arange(0.1, 0.9, 0.01)  # 0.01 step for precision
best_f1 = 0
best_thresh = 0.5

for thresh in thresholds:
    y_pred = (y_prob >= thresh).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f" 🔍 [{name}] 最佳 F1: {best_f1:.4f} | 最佳閾值: {best_thresh:.2f}")
return best_thresh, best_f1
```

### 8. Model Persistence Module

**Function**: `save_models(meta_model, best_thresh, params, iforest_model)`

**Inputs**:
- `meta_model`: Trained LogisticRegression
- `best_thresh`: Optimal threshold
- `params`: Hyperparameters dict
- `iforest_model`: Final trained Isolation Forest model

**Outputs**:
- `models/saved/f1_optimized_model.pkl`: Contains meta_model, threshold, params
- `models/saved/isolationforest.pkl`: Isolation Forest model for SHAP compatibility

**Processing**:
```python
with open(MODELS_DIR / "f1_optimized_model.pkl", "wb") as f:
    pickle.dump({
        "meta": meta_model,
        "thresh": best_thresh,
        "params": params
    }, f)

with open(MODELS_DIR / "isolationforest.pkl", "wb") as f:
    pickle.dump(iforest_model, f)
```

## Data Models

### Feature Vector

**Dimensions**: 25-28 features (exact count depends on data availability)

**Feature Categories**:
1. **User Profile** (3): age, career_code, kyc_level
2. **KYC Timing** (4): account_age_days, kyc_l1_to_l2_days, confirmed_to_l2_days, kyc_rushed
3. **TWD Transfers** (8): deposit/withdraw counts and totals, ratios, fast exit, night ratio, round amounts, active days
4. **IP Behavior** (2): ip_unique_count, ip_diversity_ratio
5. **Crypto Transfers** (3): total_count, inout_ratio, night_ratio
6. **Trading** (4): trade_count, buy_ratio, night_ratio, swap_count
7. **Cross-Table** (4): twd_to_crypto_gap_hr, activity_density, instant_wash_risk, low_kyc_high_vol
8. **Graph Features** (5): graph_degree, graph_2hop_count, graph_comp_size, graph_clustering, graph_pagerank

**Data Type**: All features are numerical (float64 or int64 after scaling)

**Scaling**: RobustScaler applied (median centering, IQR scaling)

### Model Predictions

**Base Model Outputs**:
- **LightGBM**: `predict_proba()[:, 1]` → [0,1] probability
- **XGBoost**: `predict_proba()[:, 1]` → [0,1] probability
- **Isolation Forest**: `decision_function()` → calibrated to [0,1] anomaly score

**Stacking Matrix**:
```python
X_stack = np.array([
    [lgbm_prob_1, xgb_prob_1, iforest_score_1],
    [lgbm_prob_2, xgb_prob_2, iforest_score_2],
    ...
])
```
Shape: (n_samples, 3)

**Meta-Model Output**:
- `predict_proba()[:, 1]` → [0,1] final fraud probability

**Final Prediction**:
- Binary classification: `(final_prob >= best_thresh).astype(int)`
- 0 = normal user, 1 = fraud/blacklist

### Model Artifacts

**f1_optimized_model.pkl**:
```python
{
    "meta": LogisticRegression instance,
    "thresh": float (optimal threshold),
    "params": {
        "lgbm": dict,
        "xgb": dict,
        "iforest": dict
    }
}
```

**isolationforest.pkl**:
- Standalone IsolationForest model
- Used by SHAP explainer for feature importance analysis
- Trained on full training set with optimal hyperparameters

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

After analyzing all acceptance criteria, I identified the following consolidation opportunities:

**Redundancy Analysis**:
1. Criteria 1.1, 1.2, 1.3 all verify ensemble composition → Can be combined into one property about ensemble structure
2. Criteria 3.2-3.5 all verify hyperparameter ranges → Can be combined into one property about search space validity
3. Criteria 6.1-6.5 all verify reporting behavior → These are examples of specific outputs, not properties
4. Criteria 8.1-8.4 all verify pipeline compatibility → These are integration examples, not properties
5. Criteria 9.1-9.4 all verify SHAP compatibility → These are integration examples, not properties

**Properties to Keep**:
- Feature space consistency across models (2.1)
- Anomaly score range validation (2.3)
- Stacking matrix structure (4.1)
- Test prediction aggregation (4.3)
- Score calibration range (5.1)
- Score inversion for anomalies (5.3)
- Novel fraud pattern detection (10.2)

**Examples to Keep** (not converted to properties):
- Ensemble composition verification (1.1-1.3 combined)
- Unsupervised training mode (2.2)
- Cross-validation configuration (2.4)
- Model persistence (2.5)
- Hyperparameter search space (3.1-3.7 combined)
- All reporting behaviors (6.1-6.5)
- All integration tests (8.1-8.4, 9.1-9.4)
- Novel pattern logging (10.1, 10.3, 10.4)

### Property 1: Feature Space Consistency

*For any* training run, all three base models (XGBoost, LightGBM, Isolation Forest) SHALL receive input data with the same number of features (25-28 dimensions).

**Validates: Requirements 2.1**

### Property 2: Anomaly Score Range

*For any* user in the training or test set, the calibrated Isolation Forest anomaly score SHALL be in the range [0, 1].

**Validates: Requirements 2.3, 5.1**

### Property 3: Stacking Matrix Structure

*For any* training run, the stacking feature matrix SHALL have exactly 3 columns corresponding to the three base model predictions (lgbm_oof, xgb_oof, iforest_oof).

**Validates: Requirements 4.1**

### Property 4: Test Prediction Aggregation

*For any* base model, the final test prediction SHALL be the arithmetic mean of predictions from all 5 cross-validation folds.

**Validates: Requirements 4.3**

### Property 5: Score Inversion for Anomalies

*For any* set of Isolation Forest decision_function outputs, after calibration, users with more negative decision_function values (stronger anomalies) SHALL have higher calibrated scores (closer to 1).

**Validates: Requirements 5.3**

### Property 6: Novel Fraud Pattern Detection

*For any* user where Anomaly_Score > 0.7 AND (xgb_score < 0.3 OR lgbm_score < 0.3), the system SHALL flag that user as a potential novel fraud pattern candidate.

**Validates: Requirements 10.2**

## Error Handling

### Data Loading Errors

**Missing Files**:
- If `train_balanced.csv` or `test.csv` not found, print error message and exit gracefully
- If `feature_cols.json` not found, infer feature columns from dataframe (exclude 'user_id', 'label')

**Invalid Data**:
- If feature columns contain NaN values, fill with 0 (consistent with feature engineering)
- If labels contain values other than 0/1, raise ValueError with descriptive message
- If feature dimensions mismatch between train and test, raise ValueError

### Model Training Errors

**Optuna Optimization Failures**:
- If a trial raises an exception, Optuna will catch it and continue with next trial
- If all trials fail, use default hyperparameters as fallback
- Log warning message indicating fallback to defaults

**Cross-Validation Errors**:
- If StratifiedKFold cannot split data (e.g., too few samples per class), fall back to regular KFold
- Log warning about stratification failure

**Isolation Forest Calibration Errors**:
- If all decision_function scores are identical (max_score == min_score), set all calibrated scores to 0.5
- Log warning about degenerate score distribution

### Model Persistence Errors

**Directory Creation**:
- If `models/saved/` directory doesn't exist, create it with `mkdir(parents=True, exist_ok=True)`
- If directory creation fails due to permissions, raise PermissionError with helpful message

**File Writing**:
- If pickle.dump fails, catch exception and print error message
- Ensure partial writes don't corrupt existing model files (use temporary file + rename pattern)

### Integration Errors

**SHAP Compatibility**:
- If SHAP explainer fails to initialize with meta-model, fall back to KernelExplainer
- Log warning about explainer fallback

**Pipeline Compatibility**:
- If run_pipeline.py calls train_model.py and it fails, catch exception and print error
- Ensure pipeline can continue with existing models if training fails

## Testing Strategy

### Dual Testing Approach

This feature requires both unit tests and property-based tests to ensure comprehensive coverage:

**Unit Tests** focus on:
- Specific examples of ensemble composition (3 models, correct types)
- Edge cases (empty data, single-class data, identical scores)
- Integration points (file I/O, SHAP compatibility, pipeline execution)
- Error conditions (missing files, invalid data, permission errors)

**Property-Based Tests** focus on:
- Universal properties that hold for all inputs (score ranges, feature dimensions, aggregation correctness)
- Randomized input generation to catch edge cases
- Invariants that must hold across all executions

### Property-Based Testing Configuration

**Library**: Use `hypothesis` for Python property-based testing

**Test Configuration**:
- Minimum 100 iterations per property test
- Each test tagged with: `# Feature: isolation-forest-integration, Property {N}: {property_text}`

**Property Test Examples**:

```python
from hypothesis import given, strategies as st
import numpy as np

# Property 2: Anomaly Score Range
@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=10, max_size=1000))
def test_anomaly_score_range(decision_scores):
    """
    Feature: isolation-forest-integration, Property 2: Anomaly score range
    For any user, calibrated Isolation Forest score SHALL be in [0, 1]
    """
    decision_array = np.array(decision_scores)
    calibrated = calibrate_iforest_scores(decision_array)
    
    assert np.all(calibrated >= 0), "Some scores below 0"
    assert np.all(calibrated <= 1), "Some scores above 1"
    assert calibrated.shape == decision_array.shape, "Shape mismatch"

# Property 4: Test Prediction Aggregation
@given(st.lists(st.lists(st.floats(min_value=0, max_value=1), min_size=100, max_size=100), 
                min_size=5, max_size=5))
def test_test_prediction_aggregation(fold_predictions):
    """
    Feature: isolation-forest-integration, Property 4: Test prediction aggregation
    For any base model, final test prediction SHALL be mean of 5 fold predictions
    """
    fold_array = np.array(fold_predictions)  # Shape: (5, 100)
    expected_mean = np.mean(fold_array, axis=0)
    
    # Simulate aggregation logic
    actual_mean = aggregate_test_predictions(fold_predictions)
    
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-10)

# Property 5: Score Inversion for Anomalies
@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=10, max_size=1000))
def test_score_inversion(decision_scores):
    """
    Feature: isolation-forest-integration, Property 5: Score inversion for anomalies
    For any set of scores, more negative decision_function SHALL map to higher calibrated scores
    """
    decision_array = np.array(decision_scores)
    calibrated = calibrate_iforest_scores(decision_array)
    
    # Find indices of two different scores
    if len(np.unique(decision_array)) < 2:
        return  # Skip if all scores identical
    
    min_idx = np.argmin(decision_array)
    max_idx = np.argmax(decision_array)
    
    # More negative (anomaly) should have higher calibrated score
    assert calibrated[min_idx] >= calibrated[max_idx], \
        f"Inversion failed: decision[{min_idx}]={decision_array[min_idx]} -> {calibrated[min_idx]}, " \
        f"decision[{max_idx}]={decision_array[max_idx]} -> {calibrated[max_idx]}"
```

### Unit Test Examples

```python
import pytest
import numpy as np
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb

def test_ensemble_composition():
    """Verify ensemble contains exactly XGBoost, LightGBM, and Isolation Forest"""
    # Train models
    X_train, y_train = load_test_data()
    models = train_ensemble(X_train, y_train)
    
    assert len(models) == 3, "Ensemble should have exactly 3 models"
    
    model_types = {type(m).__name__ for m in models}
    expected_types = {'XGBClassifier', 'LGBMClassifier', 'IsolationForest'}
    assert model_types == expected_types, f"Expected {expected_types}, got {model_types}"
    
    # Verify CatBoost is NOT present
    assert 'CatBoostClassifier' not in model_types, "CatBoost should be removed"

def test_unsupervised_training():
    """Verify Isolation Forest is trained without labels"""
    X_train = np.random.randn(100, 25)
    y_train = np.random.randint(0, 2, 100)
    
    # Mock IsolationForest to track fit() calls
    with patch.object(IsolationForest, 'fit', wraps=IsolationForest.fit) as mock_fit:
        model = train_isolation_forest(X_train, y_train, params={})
        
        # Verify fit was called with only X, not y
        call_args = mock_fit.call_args
        assert len(call_args[0]) == 1, "fit() should be called with only X_train"
        assert call_args[0][0] is X_train, "fit() should receive X_train"

def test_model_persistence():
    """Verify models are saved to correct paths"""
    X_train, y_train = load_test_data()
    train_and_save_models(X_train, y_train)
    
    # Check files exist
    assert Path("models/saved/f1_optimized_model.pkl").exists(), \
        "f1_optimized_model.pkl not found"
    assert Path("models/saved/isolationforest.pkl").exists(), \
        "isolationforest.pkl not found"
    
    # Verify content
    with open("models/saved/isolationforest.pkl", "rb") as f:
        model = pickle.load(f)
        assert isinstance(model, IsolationForest), \
            "isolationforest.pkl should contain IsolationForest instance"

def test_novel_fraud_detection():
    """Verify novel fraud pattern flagging logic"""
    # Create test data with known novel patterns
    anomaly_scores = np.array([0.8, 0.6, 0.75, 0.4])
    xgb_scores = np.array([0.2, 0.5, 0.25, 0.6])
    lgbm_scores = np.array([0.25, 0.4, 0.35, 0.7])
    
    flags = flag_novel_fraud_patterns(anomaly_scores, xgb_scores, lgbm_scores)
    
    # User 0: anomaly=0.8, xgb=0.2, lgbm=0.25 → FLAGGED
    assert flags[0] == True, "User 0 should be flagged"
    
    # User 1: anomaly=0.6, xgb=0.5, lgbm=0.4 → NOT FLAGGED (anomaly < 0.7)
    assert flags[1] == False, "User 1 should not be flagged"
    
    # User 2: anomaly=0.75, xgb=0.25, lgbm=0.35 → FLAGGED
    assert flags[2] == True, "User 2 should be flagged"
    
    # User 3: anomaly=0.4, xgb=0.6, lgbm=0.7 → NOT FLAGGED (anomaly < 0.7)
    assert flags[3] == False, "User 3 should not be flagged"

def test_empty_data_handling():
    """Verify graceful handling of empty datasets"""
    X_train = np.array([]).reshape(0, 25)
    y_train = np.array([])
    
    with pytest.raises(ValueError, match="Empty training set"):
        train_ensemble(X_train, y_train)

def test_single_class_data():
    """Verify handling of single-class datasets"""
    X_train = np.random.randn(100, 25)
    y_train = np.zeros(100)  # All class 0
    
    # Should fall back to regular KFold instead of StratifiedKFold
    with pytest.warns(UserWarning, match="Stratification failed"):
        models = train_ensemble(X_train, y_train)
    
    assert len(models) == 3, "Should still train 3 models"
```

### Integration Testing

**Pipeline End-to-End Test**:
```bash
# Test full pipeline with new ensemble
python run_pipeline.py --skip-fetch --skip-graph

# Verify outputs
test -f models/saved/f1_optimized_model.pkl || echo "FAIL: Model not saved"
test -f models/saved/isolationforest.pkl || echo "FAIL: IsolationForest not saved"

# Verify SHAP compatibility
python src/models/shap_explainer.py
test -f outputs/reports/shap_summary.png || echo "FAIL: SHAP failed"
```

**Performance Regression Test**:
- Train ensemble on standard test dataset
- Verify F1 score >= baseline (current CatBoost ensemble F1)
- Verify training time < 2x baseline (Isolation Forest should be faster than CatBoost)

### Test Coverage Goals

- **Unit Test Coverage**: > 90% of train_model.py lines
- **Property Test Coverage**: All 6 correctness properties implemented
- **Integration Test Coverage**: Full pipeline execution + SHAP compatibility
- **Edge Case Coverage**: Empty data, single class, identical scores, missing files

