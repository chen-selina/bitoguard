# BitoGuard SageMaker 推論容器
FROM python:3.9-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /opt/ml

# 複製 requirements
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製推論腳本
COPY inference.py /opt/ml/code/inference.py

# 設定環境變數
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_PROGRAM=inference.py

# SageMaker 會將模型檔案放在 /opt/ml/model
WORKDIR /opt/ml/code

# 啟動推論服務
ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
