# 第六章 系统展示

## 6.1 Web API 设计

本系统提供基于 FastAPI 的 RESTful API，支持单图和批量图像检测。

### 6.1.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ /predict     │    │/predict_batch│    │ /models      │  │
│  │  单图检测     │    │  批量检测     │    │  模型管理    │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘  │
│         │                    │                              │
│         └────────┬───────────┘                              │
│                  ▼                                          │
│         ┌────────────────┐                                  │
│         │  YOLO Model    │                                  │
│         │    Inference   │                                  │
│         └────────┬───────┘                                  │
│                  │                                          │
│         ┌────────▼────────┐                                 │
│         │  Result Parser  │                                 │
│         └────────┬────────┘                                 │
│                  │                                          │
│         ┌────────▼────────┐                                 │
│         │  JSON Response  │                                 │
│         └─────────────────┘                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.1.2 API 端点

#### POST `/predict` - 单图检测

**请求参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| file | UploadFile | 必需 | 图像文件 (jpg/png) |
| conf | float | 0.25 | 置信度阈值 [0.01, 0.99] |
| iou | float | 0.6 | NMS IoU阈值 [0.1, 0.95] |
| imgsz | int | 640 | 输入图像大小 [320, 1920] |
| model | str | default | 模型路径 |

**响应示例：**

```json
{
  "image_size": {"width": 640, "height": 480},
  "num_detections": 3,
  "detections": [
    {
      "class_id": 5,
      "class_name": "scratches",
      "confidence": 0.8923,
      "bbox_xyxy": [120.5, 80.2, 250.8, 180.5]
    },
    {
      "class_id": 2,
      "class_name": "patches",
      "confidence": 0.7561,
      "bbox_xyxy": [300.0, 200.0, 450.0, 350.0]
    }
  ],
  "model_path": "experiments/stage4_overall/weights/best-cosine.pt",
  "filename": "test_image.jpg",
  "annotated_image_base64": "..."
}
```

#### POST `/predict_batch` - 批量检测

**请求参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| files | list[UploadFile] | 必需 | 多张图像文件 |
| conf | float | 0.25 | 置信度阈值 |
| iou | float | 0.6 | NMS IoU阈值 |
| model | str | default | 模型路径 |

**响应示例：**

```json
{
  "model_path": "experiments/stage4_overall/weights/best-cosine.pt",
  "run_id": "20260331_143200_a1b2c3",
  "output_dir": "output/webapp_batch_outputs/20260331_143200_a1b2c3",
  "total_files": 10,
  "success_count": 9,
  "failure_count": 1,
  "results": [
    {
      "filename": "image1.jpg",
      "num_detections": 2,
      "detections": [...],
      "saved_image": "output/webapp_batch_outputs/.../image1.jpg"
    }
  ]
}
```

### 6.1.3 启动服务

```bash
cd E:\PycharmProjects\traffic-defect-detection
python -m src.webapp.app
# 或
uvicorn src.webapp.app:app --host 127.0.0.1 --port 8000
```

访问地址：`http://127.0.0.1:8000`

---

## 6.2 单图检测演示

### 6.2.1 使用示例

```python
import requests

# 上传图像进行检测
url = "http://127.0.0.1:8000/predict"

with open("test_image.jpg", "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    data = {"conf": 0.25, "iou": 0.6}

    response = requests.post(url, files=files, data=data)
    result = response.json()

print(f"检测到 {result['num_detections']} 个缺陷")
for det in result["detections"]:
    print(f"  - {det['class_name']}: {det['confidence']:.2%}")
```

### 6.2.2 返回结果说明

- **image_size**: 原始图像尺寸
- **num_detections**: 检测到的缺陷数量
- **detections**: 检测详情列表
  - class_id: 类别ID (0-5)
  - class_name: 类别名称
  - confidence: 置信度 (0-1)
  - bbox_xyxy: 边界框坐标 [x1, y1, x2, y2]
- **annotated_image_base64**: 带标注的图像（Base64编码）

---

## 6.3 批量检测演示

### 6.3.1 使用示例

```python
import requests

# 批量检测
url = "http://127.0.0.1:8000/predict_batch"

files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("image3.jpg", "rb")),
]

data = {"conf": 0.25}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"成功: {result['success_count']}/{result['total_files']}")
print(f"输出目录: {result['output_dir']}")
```

### 6.3.2 输出结构

```
output/webapp_batch_outputs/20260331_143200_a1b2c3/
├── image1.jpg          # 带标注的图像
├── image2.jpg
├── image3.jpg
└── summary.json        # 完整结果摘要
```

---

## 6.4 实际应用场景

### 6.4.1 工业生产线集成

```
┌─────────────────────────────────────────────────────────────┐
│                      工业相机                                │
│                   (实时采集钢材图像)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   边缘计算设备                                │
│              (部署YOLOv8s检测模型)                            │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  API Server (FastAPI)                               │   │
│   │  - 接收图像                                          │   │
│   │  - 执行检测                                          │   │
│   │  - 返回结果                                          │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   质检控制系统                                │
│   - 记录检测结果                                            │
│   - 触发分拣指令                                            │
│   - 生成质检报告                                            │
└─────────────────────────────────────────────────────────────┘
```

### 6.4.2 质检报告生成

检测完成后，系统可生成详细的质检报告：

```json
{
  "report_id": "QC-20260331-001",
  "timestamp": "2026-03-31T14:32:00",
  "total_inspected": 100,
  "defect_summary": {
    "crazing": 5,
    "inclusion": 8,
    "patches": 12,
    "pitted_surface": 3,
    "rolled-in_scale": 7,
    "scratches": 15
  },
  "pass_rate": "50%",
  "images_with_defects": [
    {"filename": "img_001.jpg", "defects": ["scratches", "patches"]},
    {"filename": "img_002.jpg", "defects": ["inclusion"]}
  ]
}
```

### 6.4.3 部署建议

| 环境 | 推荐配置 |
|------|---------|
| 开发测试 | CPU推理，本地FastAPI |
| 小规模部署 | 单GPU，batch_size=4 |
| 大规模生产 | 多GPU集群，模型量化 |

---

## 6.5 本章小结

本章展示了钢材表面缺陷检测系统的Web API设计与实现：

1. **RESTful API**：基于FastAPI提供标准HTTP接口
2. **单图检测**：支持实时单张图像检测
3. **批量检测**：支持多张图像批量处理
4. **模型管理**：支持动态切换不同模型

该系统可作为工业质检系统的核心组件，支持与现有MES系统集成。
