"""
test_endpoint.py
測試 SageMaker Endpoint

使用方式：
    python deploy/test_endpoint.py --endpoint-name bitoguard-endpoint
"""

import argparse
import json
import boto3
import numpy as np
import pandas as pd
from pathlib import Path


def test_endpoint(endpoint_name, region="us-west-2"):
    """
    測試 SageMaker Endpoint
    """
    print(f"\n🧪 測試 Endpoint：{endpoint_name}")
    
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    
    # 載入測試資料
    test_path = Path("data/processed/test.csv")
    if not test_path.exists():
        print("❌ 找不到測試資料：data/processed/test.csv")
        return
    
    test_df = pd.read_csv(test_path)
    
    # 載入特徵欄位
    feat_cols_path = Path("models/saved/feature_cols.json")
    with open(feat_cols_path, "r") as f:
        feature_cols = json.load(f)
    
    # 取前 5 筆測試
    X_test = test_df[feature_cols].head(5).values
    
    print(f"\n  測試資料形狀：{X_test.shape}")
    
    # 準備請求
    payload = {
        "features": X_test.tolist()
    }
    
    # 呼叫 Endpoint
    print(f"\n  呼叫 Endpoint...")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    
    # 解析結果
    result = json.loads(response["Body"].read().decode())
    
    print(f"\n✅ 預測結果：")
    print(f"  閾值：{result['threshold']}")
    print(f"\n  預測標籤：{result['predictions']}")
    print(f"  風險分數：{[round(s, 4) for s in result['risk_scores']]}")
    
    # 如果有真實標籤，顯示對比
    if "label" in test_df.columns:
        y_true = test_df["label"].head(5).values
        y_pred = result["predictions"]
        
        print(f"\n  真實標籤：{y_true.tolist()}")
        print(f"  預測標籤：{y_pred}")
        print(f"  準確度：{sum(y_true == y_pred) / len(y_true):.2%}")


def main():
    parser = argparse.ArgumentParser(description="測試 SageMaker Endpoint")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint 名稱")
    parser.add_argument("--region", default="us-west-2", help="AWS 區域")
    
    args = parser.parse_args()
    
    try:
        test_endpoint(args.endpoint_name, args.region)
    except Exception as e:
        print(f"\n❌ 測試失敗：{e}")


if __name__ == "__main__":
    main()
