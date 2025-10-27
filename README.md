signavatar-web/
# 🤖 手语识别与学习系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.x-orange.svg)](https://mindspore.cn)

基于 FastAPI 的手语识别与学习一体化平台，支持连续/孤立手语识别、系统化学习训练以及 AI 辅助教学，面向需要快速搭建手语教育与沟通工具的开发者团队。

## 🌟 功能亮点
- 🎥 **连续手语识别**：集成 Mind-VAC CSLR 管线，可批量识别视频或通过 WebSocket 处理实时关键点流，自动生成字幕与自然语言转写。
- ✋ **孤立手语识别**：基于 mind_wl I3D 模型的 Top-K 预测，支持上传短视频并反馈练习建议。
- 📚 **学习训练平台**：内置课程、任务、成就与学习路径，提供进度统计和个性化推荐。
- � **认证与安全**：支持注册/登录、刷新令牌、速率限制与安全响应头，默认使用 SQLite 存储，亦可接驳其他数据库与 Redis。
- 🤖 **AI 教学助手**：可选接入通义千问（DashScope）完成手语问答与学习资源搜索，需配置 `DASHSCOPE_API_KEY`。

## 📁 项目结构
```
.
├── backend/
│   ├── api/                # FastAPI 路由 (auth, learning, mindvac, system, websocket 等)
│   ├── core/               # 配置、服务管理、响应模型
│   ├── services/           # 识别、学习、LLM 等业务服务
│   ├── utils/              # 配置、日志、数据库、文件工具
│   └── main.py             # FastAPI 入口 (python backend/main.py)
├── frontend/               # Vite + React 前端
├── mind_vac/               # Mind-VAC 推理脚本与资源
├── mind_wl/                # 孤立手语识别模型相关文件
├── models/                 # 其他可选模型与权重
├── requirements.txt
└── README.md
```

## ⚙️ 环境准备
- **Python**：3.8 及以上（推荐 3.10）。
- **Node.js**：18+（Vite 5 需要）。
- **MindSpore**：根据硬件自行安装 2.x 版本（用于 Mind-VAC / 帧模型，未列入 requirements）。
- **数据库/缓存**：默认使用 SQLite，可选 Redis 5+。

### 后端
1. 创建虚拟环境并安装依赖：
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    ```
2. 准备模型文件（见“模型与数据”）。
3. 启动服务：
    ```powershell
    python backend/main.py
    ```
    首次启动会初始化服务管理器及数据库，并在 `temp/`、`uploads/`、`mind_vac/output_dir/` 等目录生成所需结构。

### 前端
```powershell
cd frontend
npm install
npm run dev
```
访问 http://localhost:5173 查看页面；后端默认为 http://localhost:8000。

## 🔧 配置
系统默认读取 `.env`，若不存在会自动生成基础配置；建议根据环境创建/修改 `.env`：
```env
# 基础
SECRET_KEY="请替换为 >=32 字符的随机字符串"
DATABASE_URL=sqlite:///./data/signavatar.db

# Mind-VAC
MINDVAC_ENABLED=true
MINDVAC_DEVICE=CPU
MINDVAC_CHECKPOINT_PATH=mind_vac/slr_mindspore.ckpt
MINDVAC_DICT_PATH=mind_vac/gloss_dict.npy
MINDVAC_OUTPUT_DIR=mind_vac/output_dir

# 可选：AI 教学助手
DASHSCOPE_API_KEY=<your-dashscope-api-key>
```

常用变量说明：

| 变量名 | 作用 | 默认值 |
| --- | --- | --- |
| `SECRET_KEY` | JWT 签名密钥，系统会在缺省时生成并打印，生产环境需手动设置 | 自动生成 |
| `DATABASE_URL` | 后端数据库连接串 | `sqlite:///./data/signavatar.db` |
| `REDIS_HOST` / `REDIS_PORT` | 可选 Redis 缓存配置 | `localhost` / `6379` |
| `MINDVAC_ENABLED` | 是否启用 Mind-VAC 管线 | `true` |
| `MINDVAC_DEVICE` | Mind-VAC 推理设备 (`CPU` / `GPU` / `Ascend`) | `CPU` |
| `MINDVAC_CHECKPOINT_PATH` | Mind-VAC 权重文件路径 | `mind_vac/slr_mindspore.ckpt` |
| `MINDVAC_DICT_PATH` | Mind-VAC 词典路径 (`.npy` 或 `.json`) | `mind_vac/gloss_dict.npy` |
| `MINDVAC_USE_LLM` | 是否启用 LLM 翻译增强 | `true` |
| `DASHSCOPE_API_KEY` | 通义千问 API Key，启用 AI 教学助手和 Mind-VAC LLM 时必需 | 无默认 |
| `ISOLATED_ENABLED` | 是否启用孤立手语识别服务 | `false` |
| `ISOLATED_MODEL_PATH` | 孤立手语 I3D 权重路径（启用时必填） | — |

更多可配置项见 `backend/utils/config.py` 与 `backend/core/config_manager.py`。

## 🧠 模型与数据
- `mind_vac/slr_mindspore.ckpt`：Mind-VAC 连续手语识别权重（MindSpore）。
- `mind_vac/gloss_dict.npy`：对应词典，支持 `.npy` 或 JSON 映射。
- `mind_vac/output_dir/`：Mind-VAC 推理输出目录（无需手动创建）。
- `mind_wl/`：孤立手语识别所需权重、映射表与脚本，启用前请补齐对应文件。
- `uploads/`、`temp/`：文件上传与识别结果存放目录，系统会按需创建。

## 🔌 常用接口
- `GET /api/docs`：Swagger UI。
- `GET /api/health`：基础健康检查（包含识别/学习服务状态）。
- `GET /api/system/health`：服务级健康及性能指标。
- `POST /api/sign-recognition/upload-video`：上传视频触发 Mind-VAC 识别任务。
- `GET /api/sign-recognition/status/{task_id}`：查询识别任务进度与结果。
- `WebSocket /ws/sign-recognition`：实时关键点流推理。
- `GET /api/learning/modules`：课程/模块列表。
- `POST /api/auth/login`、`POST /api/auth/register`：用户认证。

更多示例可在 `backend/api/` 中查看路由实现。

## 🧪 开发与测试
- 后端单元测试：`pytest`
- 前端测试：`npm run test`
- 前端静态检查：`npm run lint`
- 监控指标：Prometheus 端点默认在 `http://localhost:8000/metrics`，可选配 Grafana。

日志与输出：
- 后端日志默认输出到控制台，可在 `logs/` 中追加配置。
- 识别任务结果（JSON/SRT）位于 `temp/sign_results/`。

## ❓ 常见问题
- **Mind-VAC 引擎不可用**：确认权重与词典路径存在且 MindSpore 已按官方指南安装；必要时将 `MINDVAC_ENABLED` 设置为 `false` 并使用备用管线。
- **AI 教学助手返回 503**：未设置 `DASHSCOPE_API_KEY` 或网络请求受限。
- **视频上传被拒绝**：仅支持常见视频格式 (`.mp4/.avi/.mov/.mkv/.webm`)，同时需确保文件大小未超过 `MAX_UPLOAD_SIZE`（默认 100 MB）。

## 📄 许可证
本项目采用 MIT 许可证，详见 `LICENSE`。

> 让技术连接每一个人，让沟通无障碍。