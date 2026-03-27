# Requirements Document

## Introduction

BitoGuard currently uses a supervised ensemble model combining XGBoost, LightGBM, and CatBoost for fraud detection. However, reviewers have identified a critical gap: the system cannot effectively detect novel fraud patterns that differ from historical training data. This feature replaces CatBoost with Isolation Forest, an unsupervised anomaly detection algorithm, to capture previously unseen fraud behaviors and satisfy the "unsupervised learning" requirement for academic/regulatory compliance.

## Glossary

- **Ensemble_Model**: The meta-learning system that combines predictions from multiple base models (currently XGBoost, LightGBM, CatBoost)
- **Isolation_Forest**: An unsupervised anomaly detection algorithm that identifies outliers by measuring path length in random decision trees
- **Anomaly_Score**: A continuous value [0, 1] output by Isolation Forest indicating how anomalous a data point is (higher = more anomalous)
- **Base_Model**: An individual classifier within the ensemble (XGBoost, LightGBM, or Isolation Forest)
- **OOF_Predictions**: Out-of-fold predictions generated during cross-validation for stacking
- **Meta_Model**: The second-level model (LogisticRegression) that combines base model predictions
- **Novel_Fraud_Pattern**: Fraudulent behavior that differs significantly from historical training examples
- **Feature_Space**: The 25-28 dimensional space of engineered features (transaction patterns, KYC data, graph metrics)

## Requirements

### Requirement 1: Replace CatBoost with Isolation Forest

**User Story:** As a compliance analyst, I want the system to detect novel fraud patterns that don't match historical examples, so that I can identify emerging fraud schemes before they cause significant damage.

#### Acceptance Criteria

1. THE Ensemble_Model SHALL remove CatBoost as a base model
2. THE Ensemble_Model SHALL include Isolation Forest as a base model alongside XGBoost and LightGBM
3. WHEN training the Ensemble_Model, THE System SHALL train exactly three base models: XGBoost, LightGBM, and Isolation Forest
4. THE System SHALL maintain backward compatibility with existing model artifact paths in models/saved/

### Requirement 2: Isolation Forest Training and Integration

**User Story:** As a data scientist, I want Isolation Forest to be properly trained on the feature space, so that it can identify anomalous patterns in user behavior.

#### Acceptance Criteria

1. THE Isolation_Forest SHALL be trained on the same Feature_Space as XGBoost and LightGBM (25-28 features)
2. WHEN training Isolation Forest, THE System SHALL use the training set without requiring labels (unsupervised mode)
3. THE Isolation_Forest SHALL output Anomaly_Score values between 0 and 1 for each user
4. WHEN generating OOF_Predictions, THE System SHALL use StratifiedKFold cross-validation with N_SPLITS=5 for consistency with other base models
5. THE System SHALL save the trained Isolation_Forest model to models/saved/isolationforest.pkl

### Requirement 3: Hyperparameter Optimization for Isolation Forest

**User Story:** As a machine learning engineer, I want Isolation Forest hyperparameters to be optimized, so that anomaly detection performance is maximized.

#### Acceptance Criteria

1. THE System SHALL use Optuna to optimize Isolation Forest hyperparameters
2. THE Hyperparameter_Optimizer SHALL tune n_estimators (range: 100-500)
3. THE Hyperparameter_Optimizer SHALL tune max_samples (range: 0.5-1.0 or 'auto')
4. THE Hyperparameter_Optimizer SHALL tune contamination (range: 0.01-0.3)
5. THE Hyperparameter_Optimizer SHALL tune max_features (range: 0.5-1.0)
6. WHEN evaluating hyperparameters, THE System SHALL use cross-validated F1 score as the optimization objective
7. THE Hyperparameter_Optimizer SHALL run OPTUNA_TRIALS=50 trials for consistency with other models

### Requirement 4: Stacking Integration

**User Story:** As a system architect, I want Isolation Forest predictions to be properly integrated into the stacking ensemble, so that unsupervised signals contribute to final fraud predictions.

#### Acceptance Criteria

1. WHEN creating the stacking feature matrix, THE System SHALL include three columns: lgbm_oof, xgb_oof, and iforest_oof
2. THE Meta_Model SHALL be trained on the three-column stacking matrix
3. WHEN generating test predictions, THE System SHALL average predictions across all 5 folds for each base model
4. THE System SHALL maintain the existing LogisticRegression meta-model architecture
5. THE System SHALL preserve the refined threshold optimization process (find_best_threshold_refined function)

### Requirement 5: Anomaly Score Calibration

**User Story:** As a model developer, I want Isolation Forest anomaly scores to be properly calibrated, so that they are comparable to probability outputs from XGBoost and LightGBM.

#### Acceptance Criteria

1. THE System SHALL transform Isolation Forest decision_function outputs to probability-like scores in range [0, 1]
2. WHEN transforming anomaly scores, THE System SHALL use the formula: score = (decision_function - min_score) / (max_score - min_score)
3. WHERE decision_function returns negative values for anomalies, THE System SHALL invert the scale so higher scores indicate higher fraud risk
4. THE transformed scores SHALL be used for OOF predictions and test predictions

### Requirement 6: Model Evaluation and Reporting

**User Story:** As a compliance manager, I want to see how Isolation Forest contributes to overall model performance, so that I can validate the unsupervised learning component.

#### Acceptance Criteria

1. WHEN training completes, THE System SHALL report individual F1 scores for XGBoost, LightGBM, and Isolation Forest
2. THE System SHALL report the final stacked ensemble F1 score on the test set
3. THE System SHALL report precision and recall metrics for the final ensemble
4. THE System SHALL print the optimal threshold found by find_best_threshold_refined
5. THE System SHALL display confusion matrix statistics (TP, FP, TN, FN)

### Requirement 7: Dependency Management

**User Story:** As a DevOps engineer, I want all required dependencies to be properly specified, so that the system can be deployed without missing packages.

#### Acceptance Criteria

1. THE requirements.txt file SHALL include scikit-learn>=1.3.0 (which provides IsolationForest)
2. WHERE scikit-learn is already present, THE System SHALL verify version compatibility
3. THE System SHALL not introduce new external dependencies beyond existing requirements.txt

### Requirement 8: Pipeline Integration

**User Story:** As a system operator, I want the pipeline to run without modification, so that existing workflows continue to function.

#### Acceptance Criteria

1. WHEN run_pipeline.py executes step ⑤ (train_model.py), THE System SHALL train the new three-model ensemble
2. THE System SHALL maintain compatibility with existing pipeline steps (feature engineering, graph analysis, SHAP explainer)
3. THE System SHALL preserve existing command-line interfaces and arguments
4. THE System SHALL maintain the same output file structure in models/saved/

### Requirement 9: SHAP Compatibility

**User Story:** As a risk analyst, I want SHAP explanations to work with the new ensemble, so that I can understand why users are flagged as high-risk.

#### Acceptance Criteria

1. THE saved model artifact SHALL be compatible with shap_explainer.py
2. THE Meta_Model SHALL support SHAP TreeExplainer or KernelExplainer
3. WHEN generating SHAP values, THE System SHALL explain predictions based on the three base model outputs
4. THE System SHALL maintain existing SHAP visualization formats (waterfall plots, feature importance)

### Requirement 10: Novel Fraud Pattern Detection Validation

**User Story:** As a researcher, I want to validate that Isolation Forest detects novel patterns, so that I can demonstrate the value of unsupervised learning.

#### Acceptance Criteria

1. THE System SHALL identify users with high Anomaly_Score but low supervised model scores (novel pattern candidates)
2. WHEN a user has Anomaly_Score > 0.7 AND (xgb_score < 0.3 OR lgbm_score < 0.3), THE System SHALL flag them as potential novel fraud
3. THE System SHALL log the count of novel fraud pattern candidates during training
4. THE System SHALL preserve these novel pattern flags for manual review by compliance teams
