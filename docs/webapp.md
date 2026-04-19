# 交通缺陷检测系统 — WebApp 使用指南

## 项目结构

```
src/webapp/
├── app.py              # FastAPI 入口，路由注册，健康检查
├── config.py           # JWT 密钥和算法配置
├── database.py         # MySQL 数据库连接（pymysql）
├── api/
│   ├── auth.py         # 用户注册、登录、Token 验证
│   ├── detections.py   # 单图检测 / 批量检测 / 检测记录查询
│   ├── models.py       # 自定义模型上传、列表、删除
│   └── dashboard.py    # 概览统计数据（总检测数、今日检测、趋势、分类分布）
├── models/
│   ├── user.py         # 用户表（username、password_hash）
│   ├── detection_record.py  # 检测记录表
│   └── uploaded_model.py    # 上传模型表
├── middleware/
│   └── auth.py         # JWT Bearer 鉴权中间件
└── static/             # Vue.js SPA 前端（index.html + js/）
```

---

## 环境依赖

- Python 3.10+
- MySQL 5.7+（默认连接 `127.0.0.1:3306`，数据库名 `traffic_defect`）
- Node.js（前端纯静态，无需构建，直接由 FastAPI 托管）

**Python 依赖（`requirements.txt` 包含）：**

```
fastapi
uvicorn
sqlalchemy
pymysql
python-jose[cryptography]   # JWT
passlib[bcrypt]             # 密码哈希
ultralytics                 # YOLO 推理
Pillow                      # 图像处理
numpy
```

---

## 快速启动

### 1. 准备数据库

确认 MySQL 服务运行中，执行以下 SQL 创建数据库（如果不存在）：

```sql
CREATE DATABASE traffic_defect CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

> 数据库表会在应用首次启动时由 SQLAlchemy 自动创建（`Base.metadata.create_all`）。

### 2. 配置数据库账号

数据库账号密码在 `src/webapp/database.py` 中，默认：

```python
DB_HOST = "127.0.0.1"
DB_PORT = "3306"
DB_USER = "root"
DB_PASSWORD = "115269"
DB_NAME = "traffic_defect"
```

修改方法：直接编辑 `database.py`，或通过环境变量 `DB_PASSWORD` 等覆盖。

### 3. 启动后端

```bash
cd E:\PycharmProjects\traffic-defect-detection
# 激活虚拟环境（若使用虚拟环境）
.\.venv\Scripts\python -m src.webapp.app
```

服务启动后：
- API 服务：`http://127.0.0.1:8000`
- 前端页面：`http://127.0.0.1:8000/`（自动重定向到 `/static/index.html`）
- 健康检查：`GET /health`

> 首次启动时，内置模型 `02_cbam` 会自动注册到数据库。

---

## 功能模块说明

### 认证（Auth）

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/auth/register` | POST | 注册（username ≥ 3，password ≥ 6） |
| `/api/auth/login` | POST | 登录，返回 JWT token |
| `/api/auth/me` | GET | 获取当前用户信息（需 Bearer Token） |

登录后 JWT token 存储于前端本地，每次请求在 `Authorization: Bearer <token>` Header 中携带。

### 缺陷检测（Detections）

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/detections/predict` | POST | 单图检测，图片 base64 返回 |
| `/api/detections/batch` | POST | 批量检测，结果图片输出到 `output/webapp_batch_outputs/` |
| `/api/detections` | GET | 查询检测记录（分页、筛选） |
| `/api/detections/{id}` | GET | 获取单条记录详情（含 base64 图片） |
| `/api/detections/clear` | DELETE | 清除当前用户所有记录 |

**单图检测参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `conf` | 0.25 | 置信度阈值（0.01–0.99） |
| `iou` | 0.6 | IoU 阈值（0.1–0.95） |
| `imgsz` | 640 | 输入图片尺寸（320–1920） |
| `max_det` | 300 | 最大检测数（1–3000） |
| `model_id` | null | 使用上传的自定义模型（null = 内置模型） |

**检测失败处理：** 推理过程任何异常均返回原图 + `num_detections=0`，不会弹出错误提示，前端正常显示"无缺陷"。

### 模型管理（Models）

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/models` | GET | 列出内置模型 + 当前用户上传的模型 |
| `/api/models/upload` | POST | 上传 `.pt` 模型文件到 `experiments/uploaded_models/` |
| `/api/models/{id}` | DELETE | 删除自定义模型（内置模型不可删除） |

- 文件名超过 100 字符会自动截断
- 同名文件多次上传会自动加序号后缀

### 仪表盘（Dashboard）

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/dashboard/stats` | GET | 过去 7 天每日检测趋势、缺陷类型分布、总检测数、今日检测数 |

---

## 前端说明

前端为纯静态 Vue.js SPA，无需构建步骤。文件在 `src/webapp/static/`：

```
static/
├── index.html           # 入口 HTML
└── js/
    ├── router.js        # Vue Router 配置
    ├── api/client.js    # Axios 实例，拦截器注入 Token
    ├── stores/auth.js    # 登录状态管理
    └── views/
        ├── LoginView.js     # 登录/注册
        ├── DetectView.js     # 单图/批量检测
        ├── DashboardView.js  # 概览（趋势图、分类分布）
        ├── HistoryView.js    # 检测记录历史
        └── ModelsView.js     # 模型管理
```

依赖库（`static/lib/`）：Vue 3、Vue Router、Axios、Chart.js（UMD）、Tailwind CSS。

---

## 数据库表结构

### `users`
| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 主键 |
| username | VARCHAR(80) | 用户名，唯一 |
| password_hash | VARCHAR(255) | bcrypt 哈希密码 |

### `detection_records`
| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 主键 |
| user_id | INTEGER | 外键 → users.id |
| filename | VARCHAR(255) | 原始文件名 |
| model_name | VARCHAR(255) | 使用的模型名称 |
| conf / iou | FLOAT | 推理参数 |
| num_detections | INTEGER | 检测到的缺陷数量 |
| detections | JSON | 缺陷列表（class_name、bbox、confidence） |
| image_width / image_height | INTEGER | 图片尺寸 |
| annotated_image_base64 | TEXT | 标注图 base64（批量检测为空） |
| created_at | DATETIME | 检测时间 |

### `uploaded_models`
| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER | 主键 |
| name | VARCHAR(100) | 模型显示名 |
| path | VARCHAR(500) | 相对于项目根的路径，唯一 |
| is_builtin | BOOLEAN | 是否内置模型 |
| uploaded_by | INTEGER | 外键 → users.id（内置为 null） |
| uploaded_at | DATETIME | 上传时间 |

---

## 常见问题

**Q: 启动报错 `Table 'traffic_defect' doesn't exist`**
A: 先在 MySQL 中执行 `CREATE DATABASE traffic_defect CHARACTER SET utf8mb4;`

**Q: 上传模型报 500**
A: 检查 `experiments/uploaded_models/` 目录是否存在且有写入权限；确认 MySQL 连接正常。

**Q: 检测时弹 alert**
A: 检查 `/health` 接口是否返回 `"status": "ok"`，模型文件是否存在于 `experiments/02_cbam/weights/best.pt`

**Q: 概览统计数据全部为 0**
A: 先完成几次检测记录，统计数据从 `detection_records` 表中查询，检测记录不存在则数据为空。
