# BitoGuard SageMaker 部署指南

## 📋 前置需求

1. **AWS 憑證已設定**
   ```bash
   aws configure
   ```

2. **已訓練模型**
   ```bash
   python src/models/train_model.py
   ```
   
   確保以下檔案存在：
   - `models/saved/lightgbm.pkl`
   - `models/saved/xgboost.pkl`
   - `models/saved/feature_cols.json`
   - `models/saved/best_thresholds.json`

3. **安裝 SageMaker SDK**
   ```bash
   pip install sagemaker boto3
   ```

## 🚀 部署步驟

### 1. 部署到 SageMaker Endpoint

```bash
cd bitoguard
python deploy/deploy_sagemaker.py --endpoint-name bitoguard-endpoint
```

**參數說明：**
- `--endpoint-name`: Endpoint 名稱（預設：bitoguard-endpoint）
- `--instance-type`: 執行個體類型（預設：ml.m5.large）
- `--region`: AWS 區域（預設：us-west-2）
- `--bucket`: S3 bucket 名稱（預設自動偵測）
- `--role-arn`: IAM 角色 ARN（預設自動偵測）

**範例：**
```bash
# 使用預設設定
python deploy/deploy_sagemaker.py

# 自訂設定
python deploy/deploy_sagemaker.py \
  --endpoint-name my-bitoguard \
  --instance-type ml.m5.xlarge \
  --region us-east-1
```

### 2. 測試 Endpoint

```bash
python deploy/test_endpoint.py --endpoint-name bitoguard-endpoint
```

## 📡 使用 Endpoint

### Python 範例

```python
import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")

# 準備資料（特徵向量）
payload = {
    "features": [
        [0.5, 1.2, 3.4, ...],  # 第一筆資料
        [0.8, 2.1, 1.9, ...]   # 第二筆資料
    ]
}

# 呼叫 Endpoint
response = runtime.invoke_endpoint(
    EndpointName="bitoguard-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload)
)

# 解析結果
result = json.loads(response["Body"].read().decode())
print(result)
# {
#   "predictions": [0, 1],
#   "risk_scores": [0.3245, 0.7891],
#   "threshold": 0.5
# }
```

### cURL 範例

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name bitoguard-endpoint \
  --content-type application/json \
  --body '{"features": [[0.5, 1.2, 3.4]]}' \
  output.json

cat output.json
```

## 🔧 管理 Endpoint

### 查看 Endpoint 狀態

```bash
aws sagemaker describe-endpoint --endpoint-name bitoguard-endpoint
```

### 更新 Endpoint

```bash
# 重新訓練模型後
python deploy/deploy_sagemaker.py --endpoint-name bitoguard-endpoint
```

### 刪除 Endpoint

```bash
aws sagemaker delete-endpoint --endpoint-name bitoguard-endpoint
```

## 💰 成本估算

**ml.m5.large** (預設)
- 每小時：約 $0.115
- 每月（24/7）：約 $84

**ml.t2.medium** (開發測試)
- 每小時：約 $0.065
- 每月（24/7）：約 $47

**建議：**
- 開發/測試：使用 `ml.t2.medium`
- 生產環境：使用 `ml.m5.large` 或更大

## 🐛 常見問題

### 1. 找不到 SageMaker 角色

**錯誤：**
```
無法自動取得角色
```

**解決方法：**
1. 前往 AWS IAM Console
2. 建立新角色，選擇 "SageMaker"
3. 附加政策：`AmazonSageMakerFullAccess`
4. 使用 `--role-arn` 指定角色 ARN

### 2. Endpoint 名稱已存在

**錯誤：**
```
ResourceInUse: Endpoint already exists
```

**解決方法：**
```bash
# 刪除舊的 Endpoint
aws sagemaker delete-endpoint --endpoint-name bitoguard-endpoint

# 或使用不同名稱
python deploy/deploy_sagemaker.py --endpoint-name bitoguard-v2
```

### 3. 配額限制

**錯誤：**
```
ResourceLimitExceeded
```

**解決方法：**
1. 前往 AWS Service Quotas
2. 搜尋 "SageMaker"
3. 請求增加配額

## 📊 監控

### CloudWatch Metrics

```bash
# 查看 Endpoint 指標
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=bitoguard-endpoint \
  --start-time 2026-03-27T00:00:00Z \
  --end-time 2026-03-27T23:59:59Z \
  --period 3600 \
  --statistics Average
```

### 重要指標

- **ModelLatency**: 模型推論延遲
- **Invocations**: 呼叫次數
- **ModelInvocations**: 模型呼叫次數
- **InvocationErrors**: 錯誤次數

## 🔐 安全性

### IAM 政策範例

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint"
      ],
      "Resource": "arn:aws:sagemaker:us-west-2:*:endpoint/bitoguard-endpoint"
    }
  ]
}
```

### VPC 設定

如需在 VPC 內部署：

```python
model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="bitoguard-endpoint",
    vpc_config={
        "SecurityGroupIds": ["sg-xxxxx"],
        "Subnets": ["subnet-xxxxx"]
    }
)
```

## 📚 參考資料

- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [SageMaker 定價](https://aws.amazon.com/sagemaker/pricing/)
- [SageMaker 最佳實踐](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)
