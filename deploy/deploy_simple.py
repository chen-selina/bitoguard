"""
deploy_simple.py
簡化版部署腳本 - 不需要 Docker

這個版本會：
1. 將模型和依賴打包在一起
2. 使用 SageMaker 的 Python 容器
3. 在容器啟動時安裝依賴

使用方式：
    python deploy/deploy_simple.py --endpoint-name bitoguard-endpoint
"""

import argparse
import json
import tarfile
import boto3
import time
from pathlib import Path
from datetime import datetime


def create_model_package():
    """
    建立包含模型、推論腳本和依賴的 tar.gz
    """
    print("\n📦 打包模型和依賴...")
    
    models_dir = Path("models/saved")
    deploy_dir = Path("deploy")
    
    # 檢查必要檔案
    required_files = {
        "lightgbm.pkl": models_dir / "lightgbm.pkl",
        "xgboost.pkl": models_dir / "xgboost.pkl",
        "feature_cols.json": models_dir / "feature_cols.json",
        "best_thresholds.json": models_dir / "best_thresholds.json"
    }
    
    missing = [k for k, v in required_files.items() if not v.exists()]
    if missing:
        print(f"❌ 缺少檔案：{missing}")
        return None
    
    # 建立 tar.gz
    tar_path = deploy_dir / "model.tar.gz"
    
    with tarfile.open(tar_path, "w:gz") as tar:
        # 加入模型檔案
        for name, path in required_files.items():
            tar.add(path, arcname=name)
            print(f"  ✅ {name}")
        
        # 加入 CatBoost（如果存在）
        catboost_path = models_dir / "catboost.pkl"
        if catboost_path.exists():
            tar.add(catboost_path, arcname="catboost.pkl")
            print(f"  ✅ catboost.pkl")
        
        # 加入推論腳本（簡化版）
        inference_code = '''
import json
import pickle
import numpy as np
from scipy.stats import rankdata

# 安裝依賴（首次執行時）
import subprocess
import sys

def install_packages():
    packages = ["lightgbm", "xgboost", "catboost"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"安裝 {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install_packages()

# 重新 import
import lightgbm
import xgboost
try:
    import catboost
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

def model_fn(model_dir):
    """載入模型"""
    print(f"[INFO] 載入模型從：{model_dir}")
    
    models = {}
    
    # 載入 LightGBM
    try:
        with open(f"{model_dir}/lightgbm.pkl", "rb") as f:
            models["Lightgbm"] = pickle.load(f)
            print("[INFO] ✅ LightGBM")
    except Exception as e:
        print(f"[WARN] LightGBM: {e}")
    
    # 載入 XGBoost
    try:
        with open(f"{model_dir}/xgboost.pkl", "rb") as f:
            models["Xgboost"] = pickle.load(f)
            print("[INFO] ✅ XGBoost")
    except Exception as e:
        print(f"[WARN] XGBoost: {e}")
    
    # 載入 CatBoost
    if CATBOOST_AVAILABLE:
        try:
            with open(f"{model_dir}/catboost.pkl", "rb") as f:
                models["Catboost"] = pickle.load(f)
                print("[INFO] ✅ CatBoost")
        except Exception as e:
            print(f"[WARN] CatBoost: {e}")
    
    # 載入配置
    with open(f"{model_dir}/feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    
    with open(f"{model_dir}/best_thresholds.json", "r") as f:
        config = json.load(f)
    
    print(f"[INFO] 載入 {len(models)} 個模型，{len(feature_cols)} 個特徵")
    
    return {
        "models": models,
        "feature_cols": feature_cols,
        "threshold": config.get("Ensemble", 0.5),
        "weights": {"Lightgbm": 0.33, "Xgboost": 0.33, "Catboost": 0.34}
    }

def input_fn(request_body, content_type):
    """解析輸入"""
    if content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["features"])
    raise ValueError(f"不支援: {content_type}")

def predict_fn(input_data, model_dict):
    """執行預測"""
    models = model_dict["models"]
    threshold = model_dict["threshold"]
    weights = model_dict["weights"]
    
    if not models:
        raise ValueError("沒有可用的模型")
    
    # 預測
    prob_dict = {}
    for name, model in models.items():
        try:
            prob = model.predict_proba(input_data)[:, 1]
            prob_dict[name] = prob
        except Exception as e:
            print(f"[WARN] {name} 失敗: {e}")
    
    if not prob_dict:
        raise ValueError("所有模型都失敗")
    
    # Rank Ensemble
    ranked = {n: rankdata(p) / len(p) for n, p in prob_dict.items()}
    
    ensemble_prob = np.zeros(len(input_data))
    total_weight = 0
    
    for name, prob in ranked.items():
        w = weights.get(name, 1.0 / len(ranked))
        ensemble_prob += w * prob
        total_weight += w
    
    if total_weight > 0:
        ensemble_prob /= total_weight
    
    predictions = (ensemble_prob >= threshold).astype(int)
    
    return {
        "predictions": predictions.tolist(),
        "risk_scores": ensemble_prob.tolist(),
        "threshold": float(threshold)
    }

def output_fn(prediction, accept):
    """格式化輸出"""
    return json.dumps(prediction), "application/json"
'''
        
        # 寫入推論腳本
        inference_path = deploy_dir / "inference_temp.py"
        with open(inference_path, "w", encoding="utf-8") as f:
            f.write(inference_code)
        
        tar.add(inference_path, arcname="inference.py")
        print(f"  ✅ inference.py")
        
        # 清理暫存檔
        inference_path.unlink()
    
    print(f"\n✅ 打包完成：{tar_path}")
    return tar_path


def upload_to_s3(tar_path, bucket_name, prefix="bitoguard/models"):
    """上傳到 S3"""
    print(f"\n☁️  上傳到 S3...")
    
    s3_client = boto3.client("s3")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_key = f"{prefix}/model-{timestamp}.tar.gz"
    
    s3_client.upload_file(str(tar_path), bucket_name, s3_key)
    s3_uri = f"s3://{bucket_name}/{s3_key}"
    
    print(f"✅ {s3_uri}")
    return s3_uri


def deploy_endpoint(model_s3_uri, endpoint_name, instance_type, role_arn, region):
    """部署 Endpoint"""
    print(f"\n🚀 部署 SageMaker Endpoint...")
    
    sm = boto3.client("sagemaker", region_name=region)
    
    # 使用 Python 3.9 容器
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:1.13-cpu-py39"
    
    model_name = f"{endpoint_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config_name = f"{endpoint_name}-cfg-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    print(f"  模型：{model_name}")
    
    # 建立模型
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri,
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20"
            }
        },
        ExecutionRoleArn=role_arn
    )
    
    # 建立配置
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": instance_type
        }]
    )
    
    # 建立或更新 Endpoint
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"  更新 Endpoint：{endpoint_name}")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    except:
        print(f"  建立 Endpoint：{endpoint_name}")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    
    # 等待就緒
    print(f"\n  ⏳ 等待部署（約 5-10 分鐘）...")
    
    while True:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        
        if status == "InService":
            print(f"\n✅ Endpoint 已就緒！")
            break
        elif status in ["Failed", "RollingBack"]:
            print(f"\n❌ 部署失敗：{status}")
            if "FailureReason" in response:
                print(f"  原因：{response['FailureReason']}")
            return False
        else:
            print(f"  狀態：{status}...", end="\r")
            time.sleep(30)
    
    return True


def get_role_arn():
    """取得 SageMaker 角色"""
    iam = boto3.client("iam")
    roles = iam.list_roles()
    
    for role in roles["Roles"]:
        if "SageMaker" in role["RoleName"] or "sagemaker" in role["RoleName"]:
            return role["Arn"]
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", default="bitoguard-endpoint")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--bucket", default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🛡️  BitoGuard 簡化部署（無需 Docker）")
    print("=" * 60)
    
    # 1. 打包
    tar_path = create_model_package()
    if not tar_path:
        return
    
    # 2. 取得 bucket
    if args.bucket:
        bucket = args.bucket
    else:
        s3 = boto3.client("s3", region_name=args.region)
        buckets = [b["Name"] for b in s3.list_buckets()["Buckets"] if "sagemaker" in b["Name"]]
        bucket = buckets[0] if buckets else None
        
        if not bucket:
            print("❌ 找不到 SageMaker bucket")
            return
        
        print(f"\n  使用 bucket：{bucket}")
    
    # 3. 上傳
    model_uri = upload_to_s3(tar_path, bucket)
    
    # 4. 取得角色
    role_arn = get_role_arn()
    if not role_arn:
        print("❌ 找不到 SageMaker 角色")
        return
    
    print(f"\n  使用角色：{role_arn}")
    
    # 5. 部署
    success = deploy_endpoint(model_uri, args.endpoint_name, args.instance_type, role_arn, args.region)
    
    if success:
        print("\n" + "=" * 60)
        print("  🎉 部署完成！")
        print("=" * 60)
        print(f"\n  測試：python deploy/test_endpoint.py --endpoint-name {args.endpoint_name}")


if __name__ == "__main__":
    main()
