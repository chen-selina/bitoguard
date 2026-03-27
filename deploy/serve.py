"""
serve.py
SageMaker 推論服務器

這個腳本啟動一個 Flask 服務器來處理 SageMaker 的推論請求
"""

import os
import json
import flask
import traceback
from inference import model_fn, input_fn, predict_fn, output_fn

# 全域變數存放載入的模型
model = None

app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """
    健康檢查端點
    """
    global model
    
    if model is None:
        try:
            model = model_fn("/opt/ml/model")
            
            # 檢查模型是否成功載入
            if not model.get("models"):
                return flask.Response(
                    response="模型載入失敗：沒有可用的模型",
                    status=500
                )
        except Exception as e:
            return flask.Response(
                response=f"模型載入失敗：{str(e)}",
                status=500
            )
    
    return flask.Response(response="OK", status=200)


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    推論端點
    """
    global model
    
    # 確保模型已載入
    if model is None:
        try:
            model = model_fn("/opt/ml/model")
        except Exception as e:
            return flask.Response(
                response=json.dumps({"error": f"模型載入失敗：{str(e)}"}),
                status=500,
                mimetype="application/json"
            )
    
    try:
        # 解析輸入
        content_type = flask.request.content_type
        data = input_fn(flask.request.data, content_type)
        
        # 執行預測
        prediction = predict_fn(data, model)
        
        # 格式化輸出
        accept = flask.request.accept_mimetypes.best or "application/json"
        response_body, response_type = output_fn(prediction, accept)
        
        return flask.Response(
            response=response_body,
            status=200,
            mimetype=response_type
        )
    
    except Exception as e:
        error_msg = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return flask.Response(
            response=json.dumps(error_msg),
            status=500,
            mimetype="application/json"
        )


if __name__ == "__main__":
    # SageMaker 使用 port 8080
    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", 8080))
    app.run(host="0.0.0.0", port=port)
