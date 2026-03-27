"""
inference.py
SageMaker 推論腳本

這個腳本會被 SageMaker 用來載入模型並進行預測
"""

import json
import pickle
import os
import numpy as np
from pathlib import Path
from scipy.stats import rankdata


def model_fn(model_dir):
    """
    載入模型（SageMaker 會自動呼叫）
    
    Args:
        model_dir: SageMaker 解壓縮模型的目錄
    
    Returns:
        dict: 包含所有模型和配置的字典
    """
    print(f"[INFO] 載入模型從：{model_dir}")
    print(f"[INFO] 目錄內容：{os.listdir(model_dir)}")
    
    model_dir = Path(model_dir)
    
    models = {}
    model_files = ["lightgbm.pkl", "xgboost.pkl", "catboost.pkl"]
    
    for model_file in model_files:
        model_path = model_dir / model_file
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    model_name = model_file.replace(".pkl", "").capitalize()
                    models[model_name] = pickle.load(f)
                    print(f"[INFO] ✅ 載入模型：{model_name}")
            except Exception as e:
                print(f"[WARN] 無法載入 {model_file}: {e}")
    
    # 載入特徵欄位
    with open(model_dir / "feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    print(f"[INFO] 特徵數量：{len(feature_cols)}")
    
    # 載入閾值和權重
    with open(model_dir / "best_thresholds.json", "r") as f:
        config = json.load(f)
    
    print(f"[INFO] 模型載入完成，共 {len(models)} 個模型")
    
    return {
        "models": models,
        "feature_cols": feature_cols,
        "threshold": config.get("ensemble_threshold", config.get("Ensemble", 0.5)),
        "weights": config.get("weights", {})
    }


def input_fn(request_body, content_type):
    """
    解析輸入資料（SageMaker 會自動呼叫）
    
    Args:
        request_body: HTTP 請求的 body
        content_type: Content-Type header
    
    Returns:
        解析後的資料
    """
    print(f"[INFO] 收到請求，content_type: {content_type}")
    
    if content_type == "application/json":
        data = json.loads(request_body)
        features = np.array(data["features"])
        print(f"[INFO] 輸入資料形狀：{features.shape}")
        return features
    else:
        raise ValueError(f"不支援的 content_type: {content_type}")


def predict_fn(input_data, model_dict):
    """
    執行預測（SageMaker 會自動呼叫）
    
    Args:
        input_data: 從 input_fn 回傳的資料
        model_dict: 從 model_fn 回傳的模型字典
    
    Returns:
        預測結果
    """
    print(f"[INFO] 開始預測，資料形狀：{input_data.shape}")
    
    models = model_dict["models"]
    threshold = model_dict["threshold"]
    weights = model_dict["weights"]
    
    print(f"[INFO] 使用 {len(models)} 個模型，閾值：{threshold}")
    
    # 每個模型預測機率
    prob_dict = {}
    for name, model in models.items():
        try:
            prob = model.predict_proba(input_data)[:, 1]
            prob_dict[name] = prob
            print(f"[INFO] {name} 預測完成")
        except Exception as e:
            print(f"[WARN] {name} 預測失敗：{e}")
    
    if not prob_dict:
        raise ValueError("所有模型預測都失敗")
    
    # Rank Ensemble
    ranked_probs = {name: rankdata(prob) / len(prob) for name, prob in prob_dict.items()}
    
    ensemble_prob = np.zeros(len(input_data))
    
    # 如果有權重就用權重，否則平均
    if weights:
        for name, weight in weights.items():
            if name in ranked_probs:
                ensemble_prob += weight * ranked_probs[name]
    else:
        # 平均所有模型
        for prob in ranked_probs.values():
            ensemble_prob += prob / len(ranked_probs)
    
    # 套用閾值
    predictions = (ensemble_prob >= threshold).astype(int)
    
    print(f"[INFO] 預測完成，{sum(predictions)} 個高風險")
    
    return {
        "predictions": predictions.tolist(),
        "risk_scores": ensemble_prob.tolist(),
        "threshold": float(threshold)
    }


def output_fn(prediction, accept):
    """
    格式化輸出（SageMaker 會自動呼叫）
    
    Args:
        prediction: 從 predict_fn 回傳的結果
        accept: Accept header
    
    Returns:
        格式化的回應
    """
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction), "application/json"
    else:
        raise ValueError(f"不支援的 accept type: {accept}")
