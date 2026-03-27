# Implementation Plan: Isolation Forest Integration

## Overview

This plan replaces CatBoost with Isolation Forest in BitoGuard's ensemble fraud detection system. The implementation follows a sequential approach: first adding Isolation Forest training and calibration logic, then integrating it into the cross-validation pipeline, updating the stacking architecture, and finally adding novel fraud pattern detection capabilities.

## Tasks

- [ ] 1. Add Isolation Forest hyperparameter optimization
  - Add Isolation Forest parameter ranges to tune_hyperparams_v2 function
  - Support model_type="iforest" with parameters: n_estimators (100-500), max_samples (0.5-1.0), contamination (0.01-0.3), max_features (0.5-1.0)
  - Use 3-fold StratifiedKFold for inner CV consistency
  - Optimize for F1 score using calibrated anomaly scores
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ]* 1.1 Write property test for hyperparameter search space
  - **Property 1: Feature Space Consistency**
  - **Validates: Requirements 2.1**

- [ ] 2. Implement anomaly score calibration module
  - [ ] 2.1 Create calibrate_iforest_scores function
    - Accept decision_function outputs (can be negative)
    - Apply min-max normalization: (score - min) / (max - min + 1e-9)
    - Invert scale: 1 - normalized (higher = more anomalous)
    - Return calibrated scores in [0, 1] range
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 2.2 Write property test for anomaly score range
    - **Property 2: Anomaly Score Range**
    - **Validates: Requirements 2.3, 5.1**

  - [ ]* 2.3 Write property test for score inversion
    - **Property 5: Score Inversion for Anomalies**
    - **Validates: Requirements 5.3**

- [ ] 3. Integrate Isolation Forest into cross-validation pipeline
  - [ ] 3.1 Update main() to call tune_hyperparams_v2 for Isolation Forest
    - Add iforest_params = tune_hyperparams_v2(X_train, y_train, train_spw, "iforest")
    - Store params alongside lgbm_params and xgb_params
    - _Requirements: 3.1_

  - [ ] 3.2 Add Isolation Forest to 5-fold cross-validation loop
    - Initialize iforest_oof = np.zeros(len(y_train))
    - Initialize iforest_test_probs = []
    - For each fold: train IsolationForest on X_train[tr_idx] (no labels)
    - Generate decision_function scores for validation and test sets
    - Apply calibrate_iforest_scores to both validation and test scores
    - Store calibrated scores in iforest_oof[val_idx] and iforest_test_probs
    - _Requirements: 2.1, 2.2, 2.4, 5.1, 5.2, 5.3_

  - [ ]* 3.3 Write property test for stacking matrix structure
    - **Property 3: Stacking Matrix Structure**
    - **Validates: Requirements 4.1**

- [ ] 4. Update stacking architecture for three-model ensemble
  - [ ] 4.1 Modify stacking matrix creation
    - Change X_stack_train to np.column_stack([lgbm_oof, xgb_oof, iforest_oof])
    - Change X_stack_test to include np.mean(iforest_test_probs, axis=0)
    - Verify meta_model receives 3-column input
    - _Requirements: 1.2, 1.3, 4.1, 4.2_

  - [ ]* 4.2 Write property test for test prediction aggregation
    - **Property 4: Test Prediction Aggregation**
    - **Validates: Requirements 4.3**

- [ ] 5. Add model persistence for Isolation Forest
  - Train final Isolation Forest on full X_train with optimal iforest_params
  - Save to models/saved/isolationforest.pkl using pickle
  - Update f1_optimized_model.pkl to include iforest_params in params dict
  - _Requirements: 2.5, 4.4_

- [ ] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Add novel fraud pattern detection
  - [ ] 7.1 Implement flag_novel_fraud_patterns function
    - Accept anomaly_scores, xgb_scores, lgbm_scores arrays
    - Flag users where anomaly_score > 0.7 AND (xgb_score < 0.3 OR lgbm_score < 0.3)
    - Return boolean array of flags
    - _Requirements: 10.1, 10.2_

  - [ ] 7.2 Integrate novel pattern detection into main()
    - Call flag_novel_fraud_patterns with iforest_oof, xgb_oof, lgbm_oof
    - Log count of flagged users: print(f"🔍 發現 {np.sum(flags)} 個潛在新型詐騙模式")
    - Save flags to outputs/reports/novel_fraud_candidates.csv with user_ids
    - _Requirements: 10.2, 10.3, 10.4_

  - [ ]* 7.3 Write property test for novel fraud detection
    - **Property 6: Novel Fraud Pattern Detection**
    - **Validates: Requirements 10.2**

- [ ] 8. Update model evaluation reporting
  - Add individual F1 score reporting for Isolation Forest OOF predictions
  - Update final report to show contributions from all three base models
  - Add confusion matrix display for final ensemble predictions
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ]* 9. Write integration tests
  - Test ensemble composition (verify 3 models: XGBoost, LightGBM, IsolationForest)
  - Test unsupervised training mode (IsolationForest.fit called with X only)
  - Test model persistence (verify isolationforest.pkl exists and loads correctly)
  - Test pipeline compatibility (run_pipeline.py executes without errors)
  - Test SHAP compatibility (shap_explainer.py works with new ensemble)
  - _Requirements: 1.1, 1.2, 1.3, 2.2, 2.5, 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3, 9.4_

- [ ] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Integration tests validate end-to-end system behavior
- All code examples use Python (matching existing codebase)
- Chinese comments and print statements follow existing conventions
