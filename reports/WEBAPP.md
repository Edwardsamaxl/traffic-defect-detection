# Web 前后端使用说明

## 1. 安装依赖

在项目虚拟环境中安装：

```powershell
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn python-multipart pillow
```

> 若你已安装可跳过。

## 2. 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.webapp.app:app --host 127.0.0.1 --port 8000
```

浏览器打开：

- http://127.0.0.1:8000/
- 健康检查：http://127.0.0.1:8000/health

## 3. 默认模型路径

后端默认加载：

`experiments/stage4_overall/weights/best-cosine.pt`

若你后续更换模型，可修改 `src/webapp/app.py` 中 `DEFAULT_MODEL_PATH`。

## 4. API 简要说明

- `GET /health`: 返回服务和模型加载状态
- `GET /models`: 返回项目内可选 `.pt` 模型列表
- `POST /models/upload`: 上传本地 `.pt` 模型到 `experiments/uploaded_models/`
- `POST /predict`: 单图检测（form-data，字段名 `file`）
  - Query 参数：`conf`、`iou`、`imgsz`、`max_det`、`model`
  - 返回检测框列表和 base64 可视化图
- `POST /predict_batch`: 批量检测（form-data，字段名 `files`，可多文件）
  - Query 参数：`conf`、`iou`、`imgsz`、`max_det`、`model`
  - 自动将打标结果图保存到 `output/webapp_batch_outputs/<run_id>/`
  - 同时输出 `summary.json` 记录本次批量结果

## 5. 前端新功能

- 单图上传模式：用于展示检测可视化
- 批量上传模式：建议直接选择文件夹，自动处理其中全部图片
- 模型选择：从项目中的 `.pt` 文件选择模型，默认模型不变
- 本地模型上传：可像上传图片一样选择本地 `.pt` 并上传后立即可选
- 阈值预设：
  - 平衡：`conf=0.25, iou=0.60`
  - 高召回：`conf=0.10, iou=0.70`
  - 高精度：`conf=0.50, iou=0.50`
  - 自定义：手动输入

## 6. 批量输出位置

- 批量每次运行都会创建一个新目录：
  - `output/webapp_batch_outputs/<run_id>/`
- 其中包含：
  - 打标后的结果图（按上传文件夹层级保存）
  - `summary.json`（本次运行统计与逐图状态）
