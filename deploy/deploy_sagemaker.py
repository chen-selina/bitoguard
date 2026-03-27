"""
deploy_sagemaker.py
部署 BitoGuard 模型到 SageMaker Endpoint

使用方式：
    python deploy/deploy_sagemaker.py --endpoint-name bitoguard-endpoint

參數：
    --endpoint-name: Endpoint 名稱（預設：bitoguard-endpoint）
    --instance-type: 執行個體類型（預設：ml.m5.large）
    --region: AWS 區域（預設：us-west-2）
"""

import argparse
import json
import tarfile
import boto3
from pathlib import Path
from datetime import datetime


def prepare_model_artifacts():
    """
    打包模型檔案成 model.tar.gz
    """
    print("\n📦 準備模型檔案...")
    
    models_dir = Path("models/saved")
    deploy_dir = Path("deploy")
    
    # 檢查必要檔案
    required_files = [
        "lightgbm.pkl",
        "xgboost.pkl", 
        "feature_cols.json",
        "best_thresholds.json"
    ]
    
    missing_files = [f for f in required_files if not (models_dir / f).exists()]
    if missing_files:
        print(f"❌ 缺少必要檔案：{missing_files}")
        print("請先執行訓練：python src/models/train_model.py")
        return None
    
    # 建立 tar.gz
    tar_path = deploy_dir / "model.tar.gz"
    deploy_dir.mkdir(exist_ok=True)
    
    with tarfile.open(tar_path, "w:gz") as tar:
        # 加入模型檔案
        for file in required_files:
            file_path = models_dir / file
            if file_path.exists():
                tar.add(file_path, arcname=file)
                print(f"  ✅ {file}")
        
        # 加入推論腳本
        inference_script = deploy_dir / "inference.py"
        if inference_script.exists():
            tar.add(inference_script, arcname="inference.py")
            print(f"  ✅ inference.py")
        
        # 可選：CatBoost
        catboost_path = models_dir / "catboost.pkl"
        if catboost_path.exists():
            tar.add(catboost_path, arcname="catboost.pkl")
            print(f"  ✅ catboost.pkl")
    
    print(f"\n✅ 模型打包完成：{tar_path}")
    return tar_path


def upload_to_s3(tar_path, bucket_name, prefix="bitoguard/models"):
    """
    上傳模型到 S3
    """
    print(f"\n☁️  上傳模型到 S3...")
    
    s3_client = boto3.client("s3")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    s3_key = f"{prefix}/model-{timestamp}.tar.gz"
    
    s3_client.upload_file(str(tar_path), bucket_name, s3_key)
    s3_uri = f"s3://{bucket_name}/{s3_key}"
    
    print(f"✅ 上傳完成：{s3_uri}")
    return s3_uri


def deploy_endpoint(model_s3_uri, endpoint_name, instance_type, role_arn, region, image_uri=None):
    """
    部署 SageMaker Endpoint
    """
    print(f"\n🚀 部署 SageMaker Endpoint...")
    print(f"  Endpoint 名稱：{endpoint_name}")
    print(f"  執行個體類型：{instance_type}")
    
    sagemaker_client = boto3.client("sagemaker", region_name=region)
    
    # 如果沒有提供自訂映像檔，使用預設的 PyTorch 容器
    if not image_uri:
        image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:1.13-cpu-py39"
    
    model_name = f"{endpoint_name}-model-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 建立模型
    print(f"\n  建立模型：{model_name}")
    print(f"  使用容器：{image_uri}")
    
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri
            }
        },
        ExecutionRoleArn=role_arn
    )
    
    # 建立 Endpoint Config
    config_name = f"{endpoint_name}-config-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"  建立 Endpoint Config：{config_name}")
    
    sagemaker_client.create_endpoint_config(
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
        # 檢查 Endpoint 是否存在
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"  更新現有 Endpoint：{endpoint_name}")
        
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except sagemaker_client.exceptions.ClientError:
        # Endpoint 不存在，建立新的
        print(f"  建立新 Endpoint：{endpoint_name}")
        
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    
    # 等待 Endpoint 就緒
    print(f"\n  ⏳ 等待 Endpoint 部署完成（這可能需要 5-10 分鐘）...")
    waiter = sagemaker_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    
    print(f"\n✅ Endpoint 部署完成！")
    print(f"  Endpoint 名稱：{endpoint_name}")
    print(f"  狀態：InService")
    
    return endpoint_name


def get_sagemaker_role():
    """
    取得 SageMaker 執行角色
    """
    try:
        iam_client = boto3.client("iam")
        
        # 嘗試找到 SageMaker 角色
        roles = iam_client.list_roles()
        for role in roles["Roles"]:
            role_name = role["RoleName"]
            if "SageMaker" in role_name or "sagemaker" in role_name:
                print(f"  找到 SageMaker 角色：{role_name}")
                return role["Arn"]
        
        print("  ⚠️  找不到 SageMaker 角色")
        return None
    
    except Exception as e:
        print(f"  ⚠️  無法自動取得角色：{e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="部署 BitoGuard 到 SageMaker")
    parser.add_argument("--endpoint-name", default="bitoguard-endpoint", help="Endpoint 名稱")
    parser.add_argument("--instance-type", default="ml.m5.large", help="執行個體類型")
    parser.add_argument("--region", default="us-west-2", help="AWS 區域")
    parser.add_argument("--bucket", default=None, help="S3 bucket 名稱（預設自動偵測）")
    parser.add_argument("--role-arn", default=None, help="IAM 角色 ARN（預設自動偵測）")
    parser.add_argument("--image-uri", default=None, help="自訂 Docker 映像檔 URI（選填）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🛡️  BitoGuard SageMaker 部署工具")
    print("=" * 60)
    
    # 1. 打包模型
    tar_path = prepare_model_artifacts()
    if not tar_path:
        return
    
    # 2. 取得 S3 bucket
    if args.bucket:
        bucket_name = args.bucket
    else:
        s3_client = boto3.client("s3", region_name=args.region)
        buckets = s3_client.list_buckets()
        sagemaker_buckets = [b["Name"] for b in buckets["Buckets"] if "sagemaker" in b["Name"].lower()]
        
        if sagemaker_buckets:
            bucket_name = sagemaker_buckets[0]
            print(f"\n  使用 S3 bucket：{bucket_name}")
        else:
            print("❌ 找不到 SageMaker bucket，請使用 --bucket 指定")
            return
    
    # 3. 上傳到 S3
    model_s3_uri = upload_to_s3(tar_path, bucket_name)
    
    # 4. 取得 IAM 角色
    role_arn = args.role_arn or get_sagemaker_role()
    if not role_arn:
        print("\n❌ 請使用 --role-arn 指定 SageMaker 執行角色")
        return
    
    print(f"\n  使用 IAM 角色：{role_arn}")
    
    # 5. 部署 Endpoint
    try:
        endpoint_name = deploy_endpoint(model_s3_uri, args.endpoint_name, args.instance_type, role_arn, args.region, args.image_uri)
        
        print("\n" + "=" * 60)
        print("  🎉 部署成功！")
        print("=" * 60)
        print(f"\n  Endpoint 名稱：{args.endpoint_name}")
        print(f"  區域：{args.region}")
        print(f"\n  測試呼叫範例：")
        print(f'  python deploy/test_endpoint.py --endpoint-name {args.endpoint_name}')
        
    except Exception as e:
        print(f"\n❌ 部署失敗：{e}")
        print("\n可能的原因：")
        print("  1. IAM 角色權限不足")
        print("  2. Endpoint 名稱已存在")
        print("  3. 配額限制")


if __name__ == "__main__":
    main()
