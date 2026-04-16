# 交通缺陷检测系统 - PRD（产品需求文档）

## 一、核心设计理念

**去中心化，开箱即用**

- 不需要管理员角色
- 用户的存在只是为了区分"我的模型/历史"，防止他人删除
- 内置模型（`02_cbam`）启动时自动可用
- 所有登录用户都可以使用所有内置模型

---

## 二、数据库设计

### 2.1 MySQL 连接
- 数据库：MySQL
- 驱动：`pymysql` + `sqlalchemy`
- **数据库只存储模型元数据**，模型文件（`.pt`）存储在磁盘

### 2.2 表结构

**users**
```
id              — 主键，自增
username        — 用户名，唯一
password_hash   — 密码（bcrypt）
created_at      — 创建时间
```

**uploaded_models**
```
id          — 主键，自增
name        — 模型显示名
path        — 模型文件磁盘路径（相对项目根目录）
is_builtin — 是否为内置模型（True/False）
uploaded_by — 上传者用户 ID（外键 users.id），内置模型为 NULL
uploaded_at — 上传时间
```

**detection_records**
```
id                      — 主键，自增
user_id                 — 用户 ID（外键 users.id）
filename                — 原始文件名
model_name              — 使用的模型名
conf                    — 置信度阈值
iou                     — IoU 阈值
num_detections          — 检测数量
detections              — JSON，检测结果列表
image_width             — 图片宽度
image_height            — 图片高度
annotated_image_base64  — 带标注的图片（Base64）
created_at              — 创建时间
```

---

## 三、模型管理

### 3.1 模型分类

| 类型 | is_builtin | uploaded_by | 可见性 | 可删除性 |
|------|-----------|-------------|--------|---------|
| 内置模型 | True | NULL | **所有登录用户** | 不可删除 |
| 用户上传 | False | 用户ID | **仅上传者** | 仅上传者 |

### 3.2 内置模型
- **路径**：`experiments/02_cbam/weights/best.pt`
- 启动时自动注册到 `uploaded_models` 表（`is_builtin=True`, `uploaded_by=NULL`）
- **所有登录用户可见，不可删除**

### 3.3 用户上传模型
- 用户在 Models 页面上传 `.pt` 文件
- 文件保存到 `experiments/uploaded_models/`
- 数据库记录 `is_builtin=False`, `uploaded_by = 当前用户ID`
- **只有上传者可见和删除**

### 3.4 模型列表（前端 ModelsView）
每行展示：
- 模型名称（`name`）
- 模型路径（`path`）
- 上传者（内置模型显示"内置"，用户上传显示用户名）
- 删除按钮（**内置模型无删除按钮**；用户上传模型仅上传者可删除）

---

## 四、检测功能

### 4.1 单图检测 `POST /api/detections/predict`
- **需要登录**
- 记录 `user_id = 当前用户ID`

### 4.2 批量检测 `POST /api/detections/batch`
- **需要登录**
- 记录 `user_id = 当前用户ID`

### 4.3 检测历史 `GET /api/detections`
- **需要登录**
- **只能看到当前用户的记录**

### 4.4 检测详情 `GET /api/detections/:id`
- **需要登录**
- 只能查看**属于自己的记录**

### 4.5 清空历史 `DELETE /api/detections/clear`
- **需要登录**
- 只能删除**当前用户自己的记录**

---

## 五、数据概览（Dashboard）

### 5.1 接口 `GET /api/dashboard/stats`
- **需要登录**
- 统计**当前用户**的近 **7 天**数据

### 5.2 返回数据结构
```json
{
  "total_detections": 42,
  "detections_today": 3,
  "by_day": [
    {"date": "2025-01-01", "count": 5},
    {"date": "2025-01-02", "count": 3},
    ...
  ],
  "by_class": [
    {"class_name": "缺陷类型1", "count": 10},
    {"class_name": "缺陷类型2", "count": 7},
    ...
  ]
}
```

### 5.3 图表类型
- **每日检测趋势**：折线图，显示近7天每天的检测次数
- **缺陷类型分布**：饼图/柱状图，显示6个缺陷类别的出现次数

---

## 六、权限设计

### 6.1 页面权限

| 页面 | 访问条件 |
|------|---------|
| `/login` `/register` | 所有人（未登录） |
| `/detect` `/history` `/dashboard` `/models` | 已登录用户 |
| `/admin` | **已移除**（不存在） |

### 6.2 模型可见性

**`GET /api/models` 过滤逻辑**:
```python
if UploadedModel.is_builtin == True:
    # 内置模型：所有登录用户可见
    pass
elif UploadedModel.uploaded_by == current_user.id:
    # 用户上传：只有上传者可见
    pass
```

### 6.3 删除权限

**`DELETE /api/models/:id`**:
- `is_builtin == True` → 403 不可删除
- `is_builtin == False` 且 `uploaded_by == current_user.id` → 可删除
- 其他情况 → 403

---

## 七、API 端点总览

### Auth（无需登录）
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 注册 |
| POST | `/api/auth/login` | 登录 |
| GET | `/api/auth/me` | 获取当前用户信息（需登录） |

### Models（需登录）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/models` | 列出当前用户可用的模型 |
| POST | `/api/models` | 上传新模型入库 |
| DELETE | `/api/models/:id` | 删除模型（仅上传者可删） |

### Detections（需登录）
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/detections/predict` | 单图检测 |
| POST | `/api/detections/batch` | 批量检测 |
| GET | `/api/detections` | 检测历史（当前用户） |
| GET | `/api/detections/:id` | 检测详情（当前用户） |
| DELETE | `/api/detections/clear` | 清空历史（当前用户） |

### Dashboard（需登录）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/dashboard/stats` | 获取当前用户近7天统计数据 |

### Users（移除）
- 移除所有 `/api/users` 相关端点

---

## 八、前端架构

### 8.1 技术栈
Vue 3 + Vue Router + Axios，自定义 CSS（B&W Light Brutalist Minimal），单页应用

### 8.2 路由
```
/login       — 登录页
/register    — 注册页
/detect     — 缺陷检测页
/history     — 检测历史
/dashboard   — 统计面板
/models      — 模型管理（上传/查看/删除）
（/admin 路由已移除）
```

### 8.3 路由守卫
```
未登录状态访问 /detect /history /dashboard /models
→ 强制跳转 /login

已登录状态访问 /login /register
→ 强制跳转 /dashboard
```

### 8.4 API Client
- `client.js`：axios 实例，带 Bearer token，自动附加到所有请求
- 401 响应：清除 localStorage，跳转登录页

---

## 九、默认模型路径
- 路径：`experiments/02_cbam/weights/best.pt`
- 服务启动时自动注册为内置模型
- 服务启动时自动预热加载
- 检测时如未指定模型，默认使用此模型
