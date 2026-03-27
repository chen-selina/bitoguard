#!/bin/bash
# BitoGuard 完整部署流程（建置 Docker + 部署到 SageMaker）

set -e

REGION="us-west-2"
IMAGE_NAME="bitoguard"
ENDPOINT_NAME="bitoguard-endpoint"

echo "🛡️  BitoGuard 完整部署流程"
echo "================================"
echo ""

# 1. 建置並推送 Docker 映像檔
echo "步驟 1/2：建置並推送 Docker 映像檔"
python deploy/build_and_push.py \
    --image-name $IMAGE_NAME \
    --region $REGION

# 取得映像檔 URI
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:latest"

echo ""
echo "步驟 2/2：部署到 SageMaker"
python deploy/deploy_sagemaker.py \
    --endpoint-name $ENDPOINT_NAME \
    --region $REGION \
    --image-uri $IMAGE_URI

echo ""
echo "✅ 完整部署完成！"
echo ""
echo "測試 Endpoint："
echo "  python deploy/test_endpoint.py --endpoint-name $ENDPOINT_NAME"
