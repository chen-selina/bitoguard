"""
build_and_push.py
建置並推送 BitoGuard Docker 映像檔到 ECR

使用方式：
    python deploy/build_and_push.py --image-name bitoguard --region us-west-2
"""

import argparse
import boto3
import subprocess
import base64
from datetime import datetime


def get_ecr_login(region):
    """
    取得 ECR 登入憑證
    """
    print("\n🔐 取得 ECR 登入憑證...")
    
    ecr_client = boto3.client("ecr", region_name=region)
    
    try:
        response = ecr_client.get_authorization_token()
        auth_data = response["authorizationData"][0]
        
        token = base64.b64decode(auth_data["authorizationToken"]).decode()
        username, password = token.split(":")
        registry = auth_data["proxyEndpoint"]
        
        print(f"  ✅ Registry: {registry}")
        return username, password, registry
    
    except Exception as e:
        print(f"  ❌ 取得憑證失敗：{e}")
        return None, None, None


def create_ecr_repository(repository_name, region):
    """
    建立 ECR repository（如果不存在）
    """
    print(f"\n📦 檢查 ECR repository: {repository_name}")
    
    ecr_client = boto3.client("ecr", region_name=region)
    
    try:
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repo_uri = response["repositories"][0]["repositoryUri"]
        print(f"  ✅ Repository 已存在：{repo_uri}")
        return repo_uri
    
    except ecr_client.exceptions.RepositoryNotFoundException:
        print(f"  建立新 repository...")
        response = ecr_client.create_repository(
            repositoryName=repository_name,
            imageScanningConfiguration={"scanOnPush": True}
        )
        repo_uri = response["repository"]["repositoryUri"]
        print(f"  ✅ Repository 已建立：{repo_uri}")
        return repo_uri


def build_docker_image(image_name, tag="latest"):
    """
    建置 Docker 映像檔
    """
    print(f"\n🐳 建置 Docker 映像檔：{image_name}:{tag}")
    
    try:
        result = subprocess.run(
            ["docker", "build", "-t", f"{image_name}:{tag}", "-f", "deploy/Dockerfile", "deploy/"],
            check=True,
            capture_output=True,
            text=True
        )
        print("  ✅ 建置完成")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 建置失敗：{e.stderr}")
        return False


def push_to_ecr(local_image, remote_uri, username, password, registry):
    """
    推送映像檔到 ECR
    """
    print(f"\n☁️  推送映像檔到 ECR...")
    
    # Docker login
    registry_url = registry.replace("https://", "")
    try:
        subprocess.run(
            ["docker", "login", "-u", username, "-p", password, registry_url],
            check=True,
            capture_output=True
        )
        print("  ✅ Docker 登入成功")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Docker 登入失敗")
        return False
    
    # Tag image
    try:
        subprocess.run(
            ["docker", "tag", local_image, remote_uri],
            check=True,
            capture_output=True
        )
        print(f"  ✅ 標記映像檔：{remote_uri}")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 標記失敗")
        return False
    
    # Push image
    try:
        subprocess.run(
            ["docker", "push", remote_uri],
            check=True
        )
        print(f"  ✅ 推送完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 推送失敗")
        return False


def main():
    parser = argparse.ArgumentParser(description="建置並推送 BitoGuard Docker 映像檔")
    parser.add_argument("--image-name", default="bitoguard", help="映像檔名稱")
    parser.add_argument("--tag", default="latest", help="映像檔標籤")
    parser.add_argument("--region", default="us-west-2", help="AWS 區域")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🛡️  BitoGuard Docker 建置工具")
    print("=" * 60)
    
    # 1. 建置 Docker 映像檔
    local_image = f"{args.image_name}:{args.tag}"
    if not build_docker_image(args.image_name, args.tag):
        return
    
    # 2. 建立 ECR repository
    repo_uri = create_ecr_repository(args.image_name, args.region)
    if not repo_uri:
        return
    
    # 3. 取得 ECR 登入憑證
    username, password, registry = get_ecr_login(args.region)
    if not username:
        return
    
    # 4. 推送到 ECR
    remote_uri = f"{repo_uri}:{args.tag}"
    if not push_to_ecr(local_image, remote_uri, username, password, registry):
        return
    
    print("\n" + "=" * 60)
    print("  🎉 建置與推送完成！")
    print("=" * 60)
    print(f"\n  映像檔 URI：{remote_uri}")
    print(f"\n  下一步：")
    print(f"  python deploy/deploy_sagemaker.py \\")
    print(f"    --endpoint-name bitoguard-endpoint \\")
    print(f"    --image-uri {remote_uri}")


if __name__ == "__main__":
    main()
